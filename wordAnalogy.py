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

    def cos(self,W,word1,word2,vocab):
        vec1= W[vocab[word1], :]
        vec2= W[vocab[word2], :]
        return np.dot(vec1,vec2)/(linalg.norm(vec1)*linalg.norm(vec2))

    def evaluate(self,filenames,prefix, W, vocab, ivocab,words,result):
        fulldata,val= [],[]
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

        for ques in fulldata:
            indices,dist=self.distance(W, vocab, ques[:3])
            val.append(ques[3]==ivocab[indices[0]])

        OOV=num_lines-len(val)
        result[filenames.replace(".txt", "")] = (np.mean(val) * 100, np.sum(val), len(val),OOV)
        return result

    def pprint(self, collections):
        print("---------Word analogy Benchmarks results---------")
        x = PrettyTable(["Test", "Score (rho)", "Not Found/Total"])
        x.align["Dataset"] = "l"
        for result in collections:
            v = []
            for k, m in result.items():
                v.append(m)
            x.add_row([v[0], v[1], v[2]])

        print(x)
        print("---------------------------------------")

    def run(self,W, vocab, ivocab,words):
        print("Word analogy test is running...............")
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
          self.evaluate(file,prefix,W,vocab,ivocab,words,results)
        semfiles=['currency','capital-common-countries','capital-world','city-in-state', 'family']
        correct_syn, correct_sem, count_syn, count_sem=0,0,0,0
        for k, v in results.items():
          if (k in semfiles):
              count_sem += v[2]
              correct_sem += v[1]
              oov1+=v[3]

              result_sem.append({"Test":k,"Score":v[0],"OOV":str(v[3])+"/"+str(v[2]+v[3])})
          else:
              count_syn += v[2]
              correct_syn += v[1]
              oov2 += v[3]

              result_syn.append({"Test": k, "Score": v[0], "OOV": str(v[3])+"/"+str(v[2]+v[3])})
        score1=float(correct_syn)/count_syn
        score2=float(correct_sem)/count_sem
        result.append({"Test": "Syn analogy","Score":score1*100,"OOV":str(oov2)+"/"+str(count_syn),"Expand":result_syn})
        result.append({ "Test": "Sem analogy", "Score": score2*100, "OOV":str(oov1)+"/"+str(count_sem),"Expand":result_sem})
        collections.extend(result_sem)
        collections.extend(result_syn)
        self.pprint(collections)
        print("---------Overall analogy score------------")
        print("Semantic analogy score: ",score2*100)
        print("Syntactic analogy score: ", score1 * 100)
        print("------------------------------------------")
        return result

    def process(self,vectors: dict):
        W_norm, vocab, ivocab, words=self.preprocess(vectors)
        out=self.run(W_norm, vocab, ivocab, words)
        return out

