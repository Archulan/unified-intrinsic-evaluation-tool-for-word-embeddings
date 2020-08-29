import io
import csv
import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
from Evaluator import Evaluator
from nltk.cluster import KMeansClusterer

class CategorizationEvaluator(Evaluator):
    def constructEmbedding(self,filepath,vectors,words,mapcate):
        labels_true=[]
        embeddings=[]
        count,found=0,0
        with io.open(filepath, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                if row['word'] in words:
                    embeddings.append(vectors[row['word']])
                    labels_true.append(mapcate[row['category']])
                    found+=1
                count+=1

        return np.array(embeddings),labels_true,str(found)+"/"+str(count)



    def mapcluster(self,filepath):
        data={}
        df = pd.read_csv(filepath, usecols=['category'])
        category=list(set(df['category']))
        for i in range (len(category)):
            data[category[i]]=i
        return data

    def cluster(self,embedding,NUM_CLUSTERS):
        kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
        assigned_clusters = kclusterer.cluster(embedding, assign_clusters=True)
        return assigned_clusters

    def evaluate(self,labels_true, labels_pred,count):
        results={}
        homo=metrics.homogeneity_score(labels_true, labels_pred)
        complete=metrics.completeness_score(labels_true, labels_pred)
        v_score=metrics.v_measure_score(labels_true, labels_pred)
        results["Test"] = "Concept categorization"
        results["Score"] = v_score*100
        results["OOV"] = count
        self.pprint(results)
        return results

    def run(self,vectors):
        datasetpath = 'concept_cate/bless.csv'
        NUM_CLUSTERS = 17
        words = list(vectors.keys())
        print("Concept categorization test is running....")
        # map category
        mapcate=self.mapcluster(datasetpath)
        #load embeddings
        embeddings, labels_true,count=self.constructEmbedding(datasetpath,vectors,words,mapcate)
        labels_pred=self.cluster(embeddings,NUM_CLUSTERS)
        result=self.evaluate(labels_true,labels_pred,count)
        return [result]

    def pprint(self,collection):
        print("Concept categorization test results......")
        for k, v in collection.items():
            print(k," : ",v)
        print("------------------------------------------")


    def process(self, vectors: dict):
        out = self.run(vectors)
        return out
