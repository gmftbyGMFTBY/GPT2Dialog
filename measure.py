from metric.metric import *
from tqdm import tqdm
import pickle


def load_file(path):
    with open(path) as f:
        corpus = []
        for line in f.readlines():
            corpus.append(line.strip().replace('<user0>', '').replace('<user1>', '').split())
    return corpus

if __name__ == "__main__":
    ref = load_file('./data/dailydialog/tgt-test.txt')
    tgt = load_file('./data/dailydialog/pred.txt')
    
    ref = ref[:len(tgt)]
    
    print(f'[!] meausrment size: {len(ref)}')
    
    # ROUGE
    rouge_sum, counter = 0, 0
    for rr, cc in list(zip(ref, tgt)):
        rouge_sum += cal_ROUGE(rr, cc)
        counter += 1
        
    print(f'[!] ROUGE: {round(rouge_sum / counter, 4)}')
    
    # BLEU 1-4
    refs, tgts = [' '.join(i) for i in ref], [' '.join(i) for i in tgt]
    bleu1_sum, bleu2_sum, bleu3_sum, bleu4_sum = cal_BLEU(refs, tgts)
    print(f'[!] BLEU(1-4): {round(bleu1_sum, 4)}/{round(bleu2_sum, 4)}/{round(bleu3_sum, 4)}/{round(bleu4_sum, 4)}')
    
    # Dist-1/2
    candidates, references = [], []
    for line1, line2 in zip(tgt, ref):
        candidates.extend(line1)
        references.extend(line2)
    distinct_1, distinct_2 = cal_Distinct(candidates)
    redistinct_1, redistinct_2 = cal_Distinct(references)
    print(f'[!] Dist-1/2: {round(distinct_1, 4)}/{round(distinct_2, 4)}')
    print(f'[!] Real-Dist-1/2: {round(redistinct_1, 4)}/{round(redistinct_2, 4)}')
          
    # embedding-based
    with open('./data/glove_embedding.pkl', 'rb') as f:
        dic = pickle.load(f)
    ea_sum, vx_sum, gm_sum, counterp = 0, 0, 0, 0
    for rr, cc in list(zip(ref, tgt)):
        ea_sum += cal_embedding_average(rr, cc, dic)
        vx_sum += cal_vector_extrema(rr, cc, dic)
        # gm_sum += cal_greedy_matching(rr, cc, dic)
        counterp += 1
    print(f'[!] EA/VX: {round(ea_sum / counterp, 4)}/{round(vx_sum / counterp, 4)}')