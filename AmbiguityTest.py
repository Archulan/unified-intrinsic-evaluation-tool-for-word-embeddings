
import numpy as np

from scipy import linalg, stats
from prettytable import PrettyTable

def generate(model):

    words=[]
    with open(model, 'r',encoding="utf8") as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            if (len(vals)==301):
                vectors[vals[0]] = [float(x) for x in vals[1:]]
                words.append(vals[0])
    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        W[vocab[word], :] = v

    # normalize each word vector to unit variance
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    W_norm = (W.T / d).T
    return (W_norm, vocab, ivocab)


def cos(W,word1,word2):
    vec1= W[vocab[word1], :]
    vec2= W[vocab[word2], :]
    return np.dot(vec1,vec2)/(linalg.norm(vec1)*linalg.norm(vec2))


def rho(vec1,vec2):
    return stats.stats.spearmanr(vec1, vec2)[0]


def evaluate(word_dict,vocab,dataset):
    result = {}
    print("Evaluate word ambiguity....")
    with open(dataset, 'r', encoding="utf8") as data:
        pred, label, found, notfound = [], [], 0, 0
        for line in data:
            datum = line.rstrip().split("\t")
            if datum[0] in vocab and datum[1] in vocab:
                found += 1
                val=cos(word_dict,datum[0], datum[1])
                pred.append(val)
                label.append(float(datum[2]))
            else:
                notfound += 1
        result["MSD-1030"] = (found, notfound, rho(label, pred) * 100)
    return result

def pprint(result):
    print("Word ambiguity test result...")
    x = PrettyTable(["Dataset", "Found", "Not Found", "Score (rho)"])
    x.align["Dataset"] = "l"
    for k, v in result.items():
        x.add_row([k, v[0], v[1], v[2]])
    print (x)
    print("---------------------------------------")

if __name__ == "__main__":
    N = 100 # number of closest words that will be shown
    model="FastText-wiki-news-300d-1M.vec"
    W, vocab, ivocab = generate(model)
    filename = '../MSD/MSD-1030.txt'

    result=evaluate(W,vocab,filename)
    pprint(result)
