import transformers
import torch
import os
import json
import nltk
import ipdb
import random
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel
import logging
from transformers.modeling_gpt2 import GPT2Config, GPT2LMHeadModel
from transformers import BertTokenizer
from os.path import join, exists
from itertools import zip_longest, chain
# from chatbot.model import DialogueGPT2Model
from dataset import MyDataset
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
from train import create_model
import torch.nn.functional as F

PAD = '[PAD]'
pad_id = 0
logger = None


def set_interact_args():
    """
    Sets up the training arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, required=False, help='生成设备')
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成的temperature')
    parser.add_argument('--topk', default=4, type=int, required=False, help='最高k选1')
    parser.add_argument('--topp', default=0, type=float, required=False, help='最高积累概率')
    parser.add_argument('--batch_size', default=4, type=int, required=False, help='最高积累概率')
    parser.add_argument('--model_config', default='config/model_config_dialogue_small.json', type=str, required=False,
                        help='模型参数')
    parser.add_argument('--log_path', default='data/interacting.log', type=str, required=False, help='interact日志存放位置')
    parser.add_argument('--voca_path', default='vocabulary/vocab_en.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--dialogue_model_path', default='dialogue_model/empchat/model_epoch168', type=str, required=False, help='对话模型路径')
    parser.add_argument('--test_data_path', default='data/empchat/src-test.txt', type=str, required=False, help='')
    parser.add_argument('--save_samples_path', default="sample/empchat", type=str, required=False, help="保存聊天记录的文件路径")
    parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False,
                        help="重复惩罚参数，若生成的对话重复性较高，可适当提高该参数")
    parser.add_argument('--seed', type=int, default=None, help='设置种子用于生成随机数，以使得训练的结果是确定的')
    parser.add_argument('--max_len', type=int, default=50, help='每个utterance的最大长度,超过指定长度则进行截断')
    parser.add_argument('--max_history_len', type=int, default=5, help="dialogue history的最大长度")
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行预测')
    return parser.parse_args()


def create_logger(args):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=args.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


def preprocess_raw_data(raw_path, token_path, tokenizer, n_ctx):
    """
    对原始语料进行处理，将原始语料转换为用于train的token id，对于每个dialogue，将其处于成如下形式"[CLS]utterance1[SEP]utterance2[SEP]utterance3[SEP]"
    :param args:
    :param tokenizer:
    :param n_ctx:GPT2模型的上下文窗口大小,对于超过n_ctx(n_ctx包括了特殊字符)的dialogue进行截断
    :return:
    """
    logger.info("tokenizing raw data,raw data path:{}, token output path:{}".format(raw_path, token_path))
    with open(raw_path, 'rb') as f:
        data = f.read().decode("utf-8").replace('<user0>', '').replace('<user1>', '').replace('__eou__', '[SEP]')
        data = data.split('\n')
        
    logger.info("there are {} dialogue in raw dataset".format(len(data)))
    with open(token_path, "w", encoding="utf-8") as f:
        for dialogue_index, dialogue in enumerate(tqdm(data)):
            dialogue_ids = [tokenizer.cls_token_id]  # 每个dialogue以[CLS]开头
            dialogue_ids.extend([tokenizer.convert_tokens_to_ids(word.lower()) for word in nltk.word_tokenize(dialogue)])
            dialogue_ids.append(tokenizer.sep_token_id)  # 每个utterance之后添加[SEP]，表示utterance结束
            # 对超过n_ctx的长度进行截断,否则GPT2模型会报错
            if len(dialogue_ids) > n_ctx:
                dialogue_ids = [dialogue_ids[0]] + dialogue_ids[-(n_ctx-1):]
            for dialogue_id in dialogue_ids:
                f.write(str(dialogue_id) + ' ')
            # 最后一条记录不添加换行符
            if dialogue_index < len(data) - 1:
                f.write("\n")
    logger.info("finish preprocessing raw data,the result is stored in {}".format(token_path))


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        # ...表示其他维度由计算机自行推断
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # 对logits进行递减排序
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def collate_fn(batch):
    """
    计算该batch中的所有sample的最长的input，并且将其他input的长度向其对齐
    :param batch:
    :return:
    """
    global pad_id
    input_ids = []
    btc_size = len(batch)
    max_input_len = 0  # 该batch中最长的input，用于该batch的数据对齐
    # 计算该batch中input的最大长度
    for btc_idx in range(btc_size):
        if max_input_len < len(batch[btc_idx]):
            max_input_len = len(batch[btc_idx])
    # 使用pad_id对小于max_input_len的input_id进行补全
    for btc_idx in range(btc_size):
        input_len = len(batch[btc_idx])
        input_ids.append(batch[btc_idx])
        input_ids[btc_idx].extend([pad_id] * (max_input_len - input_len))
    return torch.tensor(input_ids, dtype=torch.long)


def main():
    global logger
    args = set_interact_args()
    logger = create_logger(args)
    # 当用户使用GPU,并且GPU可用时
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda' if args.cuda else 'cpu'
    logger.info('using device:{}'.format(device))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    tokenizer = BertTokenizer(vocab_file=args.voca_path)
    model = GPT2LMHeadModel.from_pretrained(args.dialogue_model_path)
    model.to(device)
    model.eval()
    
    n_ctx = model.config.to_dict().get("n_ctx")
    print(f'dialogue model path load: {args.dialogue_model_path}')
        
    
    # generate the test file
    # read the file and open the writable file
    corpus = []
    print(f'========== n_ctx of the model and datasets: {n_ctx} ==========')
    with open(args.test_data_path) as f:
        for line in f.readlines():
            line = line.lower()
            line = line.strip().replace('<user0>', '').replace('<user1>', '').replace('__eou__', '[SEP]')
            corpus.append(line)
    fw = open(args.save_samples_path, 'w')
    for line in tqdm(corpus):
        input_ids = [tokenizer.cls_token_id] + tokenizer.encode(line) + [tokenizer.sep_token_id]
        if len(input_ids) > n_ctx:
            curr_input_tensor = torch.tensor([tokenizer.cls_token_id] + input_ids[-(n_ctx-1):]).long().to(device)
        else:
            curr_input_tensor = torch.tensor(input_ids).long().to(device)
        
        generated = []
        for _ in range(args.max_len):
            outputs = model(input_ids=curr_input_tensor)
            next_token_logits = outputs[0][-1, :]
            
            # for id in set(generated):
            #     next_token_logits[id] /= args.repetition_penalty
            next_token_logits = next_token_logits / args.temperature
            next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
            filtered_logits = top_k_top_p_filtering(next_token_logits,
                                                    top_k=args.topk, 
                                                    top_p=args.topp)
            next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1),
                                           num_samples=1)
            if next_token == tokenizer.sep_token_id:  # 遇到[SEP]则表明response生成结束
                break
            generated.append(next_token.item())
            curr_input_tensor = torch.cat((curr_input_tensor, next_token), dim=0)
            
        text = tokenizer.convert_ids_to_tokens(generated)
        # ipdb.set_trace()
        text = ' '.join(text)
        fw.write(f'{text}\n')
        fw.flush()
    fw.close()
    
    '''
    while True:
        try:
            text = input("user:")
            if args.save_samples_path:
                samples_file.write("user:{}\n".format(text))
            history.append(tokenizer.encode(text))
            input_ids = [tokenizer.cls_token_id]  # 每个input以[CLS]为开头

            for history_id, history_utr in enumerate(history[-args.max_history_len:]):
                input_ids.extend(history_utr)
                input_ids.append(tokenizer.sep_token_id)
            curr_input_tensor = torch.tensor(input_ids).long().to(device)
            generated = []
            # 最多生成max_len个token
            for _ in range(args.max_len):
                outputs = model(input_ids=curr_input_tensor)
                next_token_logits = outputs[0][-1, :]
                # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
                for id in set(generated):
                    next_token_logits[id] /= args.repetition_penalty
                next_token_logits = next_token_logits / args.temperature
                # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
                next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=args.topk, top_p=args.topp)
                # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                if next_token == tokenizer.sep_token_id:  # 遇到[SEP]则表明response生成结束
                    break
                generated.append(next_token.item())
                curr_input_tensor = torch.cat((curr_input_tensor, next_token), dim=0)
                # his_text = tokenizer.convert_ids_to_tokens(curr_input_tensor.tolist())
                # print("his_text:{}".format(his_text))
            history.append(generated)
            text = tokenizer.convert_ids_to_tokens(generated)
            print("chatbot:" + " ".join(text))
            if args.save_samples_path:
                samples_file.write("chatbot:{}\n".format(" ".join(text)))
        except KeyboardInterrupt:
            if args.save_samples_path:
                samples_file.close()
            break
    '''


if __name__ == '__main__':
    main()
