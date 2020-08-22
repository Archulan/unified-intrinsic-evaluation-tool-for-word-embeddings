# -*- coding: utf-8 -*-
import io
import csv
import nltk
import numpy as np
import pandas as pd
from sklearn import metrics
from nltk.cluster import KMeansClusterer

# training data
def constructEmbedding(filepath,vectors,words,mapcate):
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

    #print(np.array(embeddings).shape)
    return np.array(embeddings),labels_true,str(found)+"/"+str(count)


def loadvocab(filepath,dim):
    with io.open(filepath, 'r',encoding="utf8") as f:
        vectors = {}
        words=[]
        for line in f:
            vals = line.rstrip().split(' ')
            if (len(vals) == int(dim)+1):
                vectors[vals[0]] = [float(x) for x in vals[1:]]
                words.append(vals[0])
    return vectors,words

def mapcluster(filepath):
    data={}
    df = pd.read_csv(filepath, usecols=['category'])
    category=list(set(df['category']))
    for i in range (len(category)):
        data[category[i]]=i
    return data

def cluster(embedding,NUM_CLUSTERS):
    kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
    assigned_clusters = kclusterer.cluster(embedding, assign_clusters=True)
    return assigned_clusters

def evaluate(labels_true, labels_pred,count):
    results={}
    homo=metrics.homogeneity_score(labels_true, labels_pred)
    complete=metrics.completeness_score(labels_true, labels_pred)
    v_score=metrics.v_measure_score(labels_true, labels_pred)
    results["#"] = 6
    results["Test"] = "Concept categorization"
    results["Score"] = v_score*100
    results["OOV"] = count

    print("Concept categorization test done")
    print("---------------------------------------")
    return results
def categorize(embeddingPath,dim):
    datasetpath = 'concept_cate/bless.csv'
    NUM_CLUSTERS = 17
    print("Concept categorization test is running....")
    # map category
    mapcate=mapcluster(datasetpath)
    #load embeddings
    vectors, words=loadvocab(embeddingPath,dim)
    embeddings, labels_true,count=constructEmbedding(datasetpath,vectors,words,mapcate)
    labels_pred=cluster(embeddings,NUM_CLUSTERS)
    result=evaluate(labels_true,labels_pred,count)
    return result
