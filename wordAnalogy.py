import io
import numpy as np
from tqdm import tqdm
from scipy import linalg
from prettytable import PrettyTable

def generate(filename,dim):
    words=[]
    with tqdm(io.open(filename, 'r',encoding="utf8")) as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            if(vals[0].isalpha() and len(vals)==int(dim)+1):
                vectors[vals[0]] = [float(x) for x in vals[1:]]
                words.append(vals[0])

    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    W_norm = np.zeros((vocab_size, vector_dim))
    print(W_norm.shape)
    count=0
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        vec=np.array(v)
        count+=1
        d = (np.sum((vec)** 2, ) ** (0.5))
        norm = (vec.T / d).T
        W_norm[vocab[word], :]=norm
        #W[vocab[word], :] = v

    print("vectors loaded")
    return (W_norm, vocab, ivocab,words)

def distance(W, vocab, ivocab, input_term):
    vecs = {}

    for idx, term in enumerate(input_term):
        vecs[idx] = W[vocab[term], :]

    vec_result = vecs[1] - vecs[0] + vecs[2]

    vec_norm = np.zeros(vec_result.shape)
    d = (np.sum(vec_result ** 2,) ** (0.5))
    vec_norm = (vec_result.T / d).T

    dist = np.dot(W, vec_norm.T)

    for term in input_term:
        index = vocab[term]
        dist[index] = -np.Inf

    a = np.argsort(-dist)[:20]
    return a,dist

def cos(W,word1,word2,vocab):
    vec1= W[vocab[word1], :]
    vec2= W[vocab[word2], :]
    return np.dot(vec1,vec2)/(linalg.norm(vec1)*linalg.norm(vec2))
    
def evaluate(filenames,prefix, W, vocab, ivocab,words,result):

    results=[]
    #for i in range(len(filenames)):
    fulldata = []
    val=[]
    feed = io.open('%s/%s' % (prefix, filenames), 'r', encoding="utf8").read()
    num_lines = len(feed.splitlines())
    with io.open('%s/%s' % (prefix, filenames), 'r') as f:
        for line in tqdm(f):
            row=line.rstrip().split()
            if False in [i in words for i in row]:
                continue
            fulldata.append(row)

    if len(fulldata) == 0:
        print("ERROR: no lines of vocab kept for %s !" % filenames)
        #continue
    for ques in fulldata:
        indices,dist=distance(W, vocab, ivocab, ques[:3])
        val.append(ques[3]==ivocab[indices[0]])
        #if(ques[3]==ivocab[indices[0]]):
         #   print(ivocab[indices[0]])


    OOV=num_lines-len(val)
    result[filenames.replace(".txt", "")] = (np.mean(val) * 100, np.sum(val), len(val),OOV)
    #results.append(result)
    #pprint(result)
    return result
def analogy(filename,dim):
    print("Word analogy test is running")
    W, vocab, ivocab,words = generate(filename,dim)
    collections=[]
    filenames = [
        'currency.txt','capital-common-countries.txt','capital-world.txt',
        'city-in-state.txt', 'family.txt', 'gram1-adjective-to-adverb.txt',
        'gram2-opposite.txt', 'gram3-comparative.txt', 'gram4-superlative.txt',
        'gram5-present-participle.txt', 'gram6-nationality-adjective.txt',
        'gram7-past-tense.txt', 'gram8-plural.txt', 'gram9-plural-verbs.txt'
    ]
    prefix = 'question-data'

    results={}
    result,result_sem,result_syn=[],[],[]
    oov1,oov2=0,0
    for file in tqdm(filenames):
      evaluate(file,prefix,W,vocab,ivocab,words,results)
    semfiles=['currency','capital-common-countries','capital-world','city-in-state', 'family']
    correct_syn, correct_sem, count_syn, count_sem=0,0,0,0
    for k, v in results.items():
      if (k in semfiles):
          count_sem += v[2]
          correct_sem += v[1]
          oov1+=v[3]
          print(v[1],v[2])
          result_sem.append({"Test":k,"Score":v[0],"OOV":v[3]})
      else:
          count_syn += v[2]
          correct_syn += v[1]
          oov2 += v[3]
          print(v[1], v[2])
          result_syn.append({"Test": k, "Score": v[0], "OOV": v[3]})
    score1=float(correct_syn)/count_syn
    score2=float(correct_sem)/count_sem
    result.append({"Test": "Syn analogy","Score":score1*100,"OOV":str(oov2)+"/"+str(count_syn),"Expand":result_syn})
    result.append({ "Test": "Sem analogy", "Score": score2*100, "OOV":str(oov1)+"/"+str(count_sem),"Expand":result_sem})
    collections.extend(result_sem)
    collections.extend(result_syn)
    pprint(collections)
    print("Semantic analogy score: ",score2*100)
    print("Syntactic analogy score: ", score1 * 100)
    print("analogy test done..")
    return result


def pprint(collections):
    print("Word analogy test results")
    x = PrettyTable(["Test", "Score (rho)", "Not Found/Total"])
    x.align["Dataset"] = "l"
    for result in collections:
        v = []
        for k, m in result.items():
            v.append(m)
        x.add_row([ v[0],v[1], v[2]])

    print(x)
    print("---------------------------------------")
