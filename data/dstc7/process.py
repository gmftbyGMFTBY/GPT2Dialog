from tqdm import tqdm
import ipdb

def load_file(mode):
    path = f'dial.{mode}'
    with open(path) as f:
        corpus = []
        for line in f.readlines():
            utterances = line.strip().split('\t')
            utterances = [i.strip() for i in utterances]
            line = ' __eou__ '.join(utterances)
            corpus.append(line)
    return corpus

def write_file(corpus, path):
    with open(path, 'w') as f:
        for dialog in tqdm(corpus):
            utterances = dialog.split('__eou__')
            utterances = [i.strip() for i in utterances]
            for utterance in utterances:
                f.write(f'{utterance}\n')
            f.write('\n')



if __name__ == "__main__":
    import sys
    mode = sys.argv[1]
    data = load_file(mode)
    write_file(data, f'{mode}.txt')
