import io
import numpy as np
from tqdm import tqdm
from scipy import linalg, stats
from Evaluator import Evaluator
from prettytable import PrettyTable
from collections import defaultdict

class SimilarityEvaluator(Evaluator):
    def preprocess(self,vectors: dict):
        print("Preprocessing the vector file for similarity test")
        words=list(vectors.keys())
        vocab_size = len(words)
        vector_dim=len(list(vectors.values())[0])
        vocab = {w: idx for idx, w in enumerate(words)}

        W_norm = np.zeros((vocab_size, vector_dim))
        for word, v in vectors.items():
            if word == '<unk>':
                continue
            vec = np.array(v)
            d = (np.sum((vec) ** 2, ) ** (0.5))
            norm = (vec.T / d).T
            W_norm[vocab[word], :] = norm
        return (W_norm, vocab)



    def cos(self,W,word1,word2,vocab):
        vec1= W[vocab[word1], :]
        vec2= W[vocab[word2], :]
        return np.dot(vec1,vec2)/(linalg.norm(vec1)*linalg.norm(vec2))


    def rho(self,vec1,vec2):
        return stats.stats.spearmanr(vec1, vec2)[0]

    def evaluate(self,word_dict,vocab,dataset):
        result = []
        for file_name, data in tqdm(dataset.items()):
            pred, label, found, notfound,temp = [], [], 0, 0, {}
            for datum in data:
                if datum[0] in vocab and datum[1] in vocab:
                    found += 1
                    val=self.cos(word_dict,datum[0], datum[1],vocab)
                    pred.append(val)
                    label.append(datum[2])
                else:
                    notfound += 1
            temp["Test"]=file_name
            temp["Score"] = self.rho(label, pred) * 100
            temp["OOV"] = str(notfound) + str("/") + str(found + notfound)
            result.append(temp)
        return result


    def run( self,W, vocab):
        print("Word similarity test is Running........")
        score,notfound,total=0,0,0
        results=[]
        filenames = [
            'EN-WS-353-REL.txt','EN-WS-353-SIM.txt','EN-RW-STANFORD.txt','EN-MEN-TR-3k.txt','EN-VERB-143.txt','MSD-1030.txt'
        ]
        prefix = 'word-sim'
        dataset=defaultdict(list)
        for file_name in filenames:
            with io.open('%s/%s' % (prefix, file_name), 'r',encoding='utf8') as f:
                for line in f:
                    dataset[file_name.replace(".txt","")].append([float(w) if i == 2 else w for i, w in enumerate(line.rstrip().split())])

        result=self.evaluate(W,vocab,dataset)
        temp,temp1={},{}
        for k in result:
            if(k["Test"] == "EN-RW-STANFORD"):
                temp1 = k
                results.append({ "Test": "RW similarity", "Score": k["Score"], "OOV": k["OOV"], "Expand": []})
            elif(k["Test"] == "MSD-1030"):
                temp = k
                results.append({ "Test": "Ambiguity", "Score": k["Score"], "OOV": k["OOV"], "Expand": []})
            else:
                score += k["Score"]
                notfound += int(k["OOV"].split("/")[0])
                total += int(k["OOV"].split("/")[1])
        print("\n-------- Word similarity benchmark test results --------")
        self.pprint(result)
        result.remove(temp)
        result.remove(temp1)
        results.append({"Test":"Word similarity","Score":str(score/4),"OOV":str(notfound)+str("/")+str(total),"Expand":result})
        print("\n---------------------- Overall results ---------------------")
        self.pprint(results)

        return results

    def pprint(self,collections):

        x = PrettyTable(["Test", "Score (rho)","Not found/Total"])
        x.align["Dataset"] = "l"
        for result in collections:
            v=[]
            for k, m in result.items():
                v.append(m)
            x.add_row([v[0],v[1], v[2]])

        print (x,'\n')

    def process(self, vectors: dict):
        W_norm, vocab= self.preprocess(vectors)
        out = self.run(W_norm, vocab)
        return out

