import numpy as np
from scipy import linalg, stats
from collections import defaultdict
import io
def generate(filename,dim):

    words=[]
    with io.open(filename, 'r',encoding="utf8") as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            if (len(vals)==int(dim)+1 and vals[0].isalpha() ):
                vectors[vals[0]] = [float(x) for x in vals[1:]]
                words.append(vals[0])
    vocab_size = len(words)
    print(vocab_size)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        W[vocab[word], :] = v

    # normalize each word vector to unit variance
  
    print("generate vectors list")
    return (W, vocab, ivocab)


def distance(W, vocab, ivocab, input_term):
    for idx, term in enumerate(input_term.split(' ')):
        if term in vocab:
            #print('Word: %s  Position in vocabulary: %i' % (term, vocab[term]))
            if idx == 0:
                vec_result = np.copy(W[vocab[term], :])
            else:
                vec_result += W[vocab[term], :]
        else:
            print('Word: %s  Out of dictionary!\n' % term)
            return

    vec_norm = np.zeros(vec_result.shape)
    d = (np.sum(vec_result ** 2,) ** (0.5))
    vec_norm = (vec_result.T / d).T

    dist = np.dot(W, vec_norm.T)

    for term in input_term.split(' '):
        index = vocab[term]
        dist[index] = -np.Inf

    a = np.argsort(-dist)[:10]


def cos(W,word1,word2,vocab):
    vec1= W[vocab[word1], :]
    vec2= W[vocab[word2], :]
    return np.dot(vec1,vec2)/(linalg.norm(vec1)*linalg.norm(vec2))


def rho(vec1,vec2):
    return stats.stats.spearmanr(vec1, vec2)[0]

def evaluate(word_dict,vocab,dataset):
    result = []
    print("Evaluate word similarity....")
    for file_name, data in dataset.items():
        pred, label, found, notfound,temp = [], [], 0, 0, {}
        for datum in data:
            if datum[0] in vocab and datum[1] in vocab:
                found += 1
                val=cos(word_dict,datum[0], datum[1],vocab)
                pred.append(val)
                label.append(datum[2])
            else:
                notfound += 1
        temp["#"] = ""
        temp["Score"]=rho(label, pred) * 100
        temp["OOV"]=str(notfound)+str("/")+str(found+notfound)
        temp["Test"]=file_name
        result.append(temp)
    return result


def similarity(filename,dim):
    score,notfound,total=0,0,0
    results=[]
    W, vocab, ivocab = generate(filename,dim)
    filenames = [
        'EN-WS-353-REL.txt','EN-WS-353-SIM.txt','EN-RW-STANFORD.txt','EN-MEN-TR-3k.txt','EN-VERB-143.txt','MSD-1030.txt'
    ]
    prefix = 'c:\\users\\hp\\Desktop\\fyp\\data\\Evaluator\\flask\\app\\evaluators\\word-sim'
    dataset=defaultdict(list)
    for file_name in filenames:
        with io.open('%s/%s' % (prefix, file_name), 'r',encoding='utf8') as f:
            for line in f:
                dataset[file_name.replace(".txt","")].append([float(w) if i == 2 else w for i, w in enumerate(line.rstrip().split())])

    result=evaluate(W,vocab,dataset)
    temp,temp1={},{}
    for k in result:
        if(k["Test"] == "EN-RW-STANFORD"):
            temp1 = k
            results.append({"#": 1, "Test": "RW similarity", "Score": k["Score"], "OOV": k["OOV"], "Expand": []})
        elif(k["Test"] == "MSD-1030"):
            temp = k
            results.append({"#": 2, "Test": "Ambiguity", "Score": k["Score"], "OOV": k["OOV"], "Expand": []})
        else:
            score += k["Score"]
            notfound += int(k["OOV"].split("/")[0])
            total += int(k["OOV"].split("/")[1])
    result.remove(temp)
    result.remove(temp1)
    results.append({"#":3,"Test":"Word similarity","Score":str(score/4),"OOV":str(notfound)+str("/")+str(total),"Expand":result})
    return results
