import io
import numpy as np
from tqdm import tqdm
from scipy import linalg
from Evaluator import Evaluator
from prettytable import PrettyTable

class AnalogyEvaluator(Evaluator):
    def preprocess(self,vectors: dict):
        print("Preprocessing the vector file for Analogy test")
        words=list(vectors.keys())
        vocab_size = len(words)
        vocab = {w: idx for idx, w in enumerate(words)}
        ivocab = {idx: w for idx, w in enumerate(words)}

        vector_dim = len(vectors[ivocab[0]])
        W_norm = np.zeros((vocab_size, vector_dim))

        for word, v in vectors.items():
            if word == '<unk>':
                continue
            vec = np.array(v)
            d = (np.sum((vec) ** 2, ) ** (0.5))
            norm = (vec.T / d).T
            W_norm[vocab[word], :] = norm

        return (W_norm, vocab, ivocab, words)


    def distance(self,W, vocab, input_term):
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

    def cosmul(self, W, vocab, input_term):
        vecs = {}

        for idx, term in enumerate(input_term):
            vecs[idx] = W[vocab[term], :]

        A = np.zeros(vecs[0].shape)
        d = (np.sum(vecs[0] ** 2, ) ** (0.5))
        A = (vecs[0].T / d).T

        B = np.zeros(vecs[1].shape)
        d = (np.sum(vecs[1] ** 2, ) ** (0.5))
        B = (vecs[1].T / d).T

        C = np.zeros(vecs[2].shape)
        d = (np.sum(vecs[2] ** 2, ) ** (0.5))
        C = (vecs[2].T / d).T

        D_A = np.log((1.0 + np.dot(W, A.T)) / 2.0 + 1e-5)
        D_B = np.log((1.0 + np.dot(W, B.T)) / 2.0 + 1e-5)
        D_C = np.log((1.0 + np.dot(W, C.T)) / 2.0 + 1e-5)
        D = D_B - D_A + D_C
        for term in input_term:
            index = vocab[term]
            D[index] = np.finfo(np.float32).min
        a = D.argmax(axis=0)
        return a, D

    def cos(self,W,word1,word2,vocab):
        vec1= W[vocab[word1], :]
        vec2= W[vocab[word2], :]
        return np.dot(vec1,vec2)/(linalg.norm(vec1)*linalg.norm(vec2))

    def evaluate(self, filenames, prefix, W, vocab, ivocab, words, result):
        fulldata, val, val2 = [], [], []
        num_lines = 0
        with io.open('%s/%s' % (prefix, filenames), 'r') as f:
            for line in tqdm(f):
                row = line.rstrip().split()
                num_lines += 1
                if False in [i in words for i in row]:
                    continue
                fulldata.append(row)

        if len(fulldata) == 0:
            print("ERROR: no lines of vocab kept for %s !" % filenames)

        for ques in fulldata:
            indices,dist=self.distance(W, vocab, ques[:3])
            indices2, dist2 = self.cosmul(W, vocab, ques[:3])
            val.append(ques[3]==ivocab[indices[0]])
            val2.append(ques[3] == ivocab[indices2])

        print(len(val))
        OOV=num_lines-len(val)
        result[filenames.replace(".txt","")] = (np.mean(val) * 100, np.sum(val), len(val),OOV,np.mean(val2) * 100, np.sum(val2))
        return result

    def pprint(self, collections):
        print("---------Word analogy Benchmarks results---------")
        x = PrettyTable(["Test", "Score (rho)-cos-add","Score -cos-mul", "Not Found/Total"])
        x.align["Dataset"] = "l"
        for result in collections:
            v = []
            for k, m in result.items():
                v.append(m)
            x.add_row([v[0], v[1], v[2],v[3]])

        print(x)
        print("---------------------------------------")

    def run(self,W, vocab, ivocab,words):
        print("Word analogy test is running...............")
        collections=[]
        filenames = [
            'currency.txt', 'capital-common-countries.txt', 'capital-world.txt',
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
          self.evaluate(file,prefix,W,vocab,ivocab,words,results)
        semfiles=['currency','capital-common-countries','capital-world','city-in-state', 'family']
        correct_syn, correct_sem, count_syn, count_sem,correct_sem_mul,correct_syn_mul=0,0,0,0,0,0

        for k, v in results.items():
          if (k in semfiles):
              count_sem += v[2]
              correct_sem += v[1]
              oov1+=v[3]
              correct_sem_mul += v[5]
              result_sem.append({"Test":k,"Cos-add-Score": v[0],"Cos-mul-score":v[4],"OOV":str(v[3])+"/"+str(v[2]+v[3])})
          else:
              count_syn += v[2]
              correct_syn += v[1]
              oov2 += v[3]
              correct_syn_mul+=v[5]
              result_syn.append({"Test": k, "Cos-add-Score": v[0],"Cos-mul-score":v[4],"OOV": str(v[3])+"/"+str(v[2]+v[3])})

        Syn_score1=  float(correct_syn)/count_syn
        Sem_score1=  float(correct_sem)/count_sem
        Syn_score2 = float(correct_syn_mul) / count_syn
        Sem_score2 = float(correct_sem_mul) / count_sem
        result.append({"Test": "Syn analogy","Cos-add-Score":Syn_score1*100,"Cos-mul-Score":Syn_score2*100,"OOV":str(oov2)+"/"+str(count_syn),"Expand":result_syn})
        result.append({ "Test": "Sem analogy", "Cos-add-Score": Sem_score1*100,"Cos-mul-Score":Sem_score2*100, "OOV":str(oov1)+"/"+str(count_sem),"Expand":result_sem})
        collections.extend(result_sem)
        collections.extend(result_syn)
        self.pprint(collections)
        print("---------Overall analogy score------------")
        print("Semantic analogy score using Cos-add method: ", Sem_score1*100)
        print("Syntactic analogy score using Cos-add method:", Syn_score1* 100)
        print("Semantic analogy score using Cos-mul method: ", Sem_score2 * 100)
        print("Syntactic analogy score using Cos-mul method:", Syn_score2 * 100)
        print("------------------------------------------")
        return result

    def process(self,vectors: dict):
        W_norm, vocab, ivocab, words=self.preprocess(vectors)
        out=self.run(W_norm, vocab, ivocab, words)
        return out

