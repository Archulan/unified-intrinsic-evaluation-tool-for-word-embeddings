import os
import io
import operator
import numpy as np
from math import sqrt

from prettytable import PrettyTable


class OutlierDetectionCluster:
    # Class modeling a cluster of the dataset, composed of its topic name, its corresponding elements and the outliers to be detected
    def __init__(self, elements, outliers, topic=""):
        self.elements = elements
        self.outliers = outliers
        self.topic = topic


class OutlierDetectionDataset:
    # Class modeling a whole outlier detection dataset composed of various topics or clusters
    def __init__(self, path):
        self.path = path
        self.setWords = set()
        self.clusters = set()

    def readDataset(self):
        print("\nReading outlier detection dataset...")
        dict_cluster_elements = {}
        dict_cluster_outliers = {}
        listing = os.listdir(self.path)
        for in_file in listing:
            if in_file.endswith(".txt"):
                with io.open(self.path + in_file, 'r', encoding="utf8") as cluster_file:
                    cluster_name = in_file.replace(".txt", "")
                    set_elements = set()
                    set_outliers = set()
                    cluster_boolean = True
                    for line in cluster_file:
                        if cluster_boolean:
                            if line != "\n":
                                word = line.strip().replace(" ", "_")
                                set_elements.add(word)
                                self.setWords.add(word)
                                if "_" in word:
                                    for unigram in word.split("_"):
                                        self.setWords.add(unigram)
                            else:
                                cluster_boolean = False
                        else:
                            if line != "\n":
                                word = line.strip().replace(" ", "_")
                                set_outliers.add(word)
                                self.setWords.add(word)
                                if "_" in word:
                                    for unigram in word.split("_"):
                                        self.setWords.add(unigram)
                self.clusters.add(OutlierDetectionCluster(set_elements, set_outliers, cluster_name))


def boolean_answer(answer):
    if answer.lower() == "y" or answer.lower() == "yes":
        return True
    elif answer.lower() == "n" or answer.lower() == "no":
        return False
    else:
        new_answer = input('Please answer "Yes" or "No"')
        return boolean_answer(new_answer)


def module(vector):
    # Module of a vector
    suma = 0.0
    for dimension in vector:
        suma += dimension * dimension
    return sqrt(suma)


def scalar_prod(vector1, vector2):
    # Scalar product between two vectors
    prod = 0.0
    for i in range(len(vector1)):
        dimension_1 = vector1[i]
        dimension_2 = vector2[i]
        prod += dimension_1 * dimension_2
    return prod


def cosine(vector1, vector2):
    # Cosine similarity between two vectors
    module_vector_1 = module(vector1)
    if module_vector_1 == 0.0: return 0.0
    module_vector_2 = module(vector2)
    if module_vector_2 == 0.0: return 0.0
    return scalar_prod(vector1, vector2) / (module(vector1) * module(vector2))


def pairwisesimilarities_cluster(setElementsCluster, input_vectors):
    # This function calculates all pair-wise similarities between the elements of a cluster and stores them in a dictionary
    dict_sim = {}
    for element_1 in setElementsCluster:
        for element_2 in setElementsCluster:
            if element_1 != element_2:
                dict_sim[element_1 + " " + element_2] = cosine(input_vectors[element_1], input_vectors[element_2])
    return dict_sim


def compose_vectors_multiword(multiword,input_vectors,dimensions):
    #Given an OOV word as input, this function either returns a vector by averaging the vectors of each token composing a multiword expression or a zero vector
    vector_multiword=[0.0]*dimensions
    cont_unigram_in_vectors=0
    for unigram in multiword.split("-"):
        if unigram in input_vectors:
            cont_unigram_in_vectors+=1
            vector_unigram=input_vectors[unigram]
            for i in range(dimensions):
                vector_multiword[i]+=vector_unigram[i]
    if cont_unigram_in_vectors>0:
        for j in range(dimensions):
            vector_multiword[j]=vector_multiword[j]/cont_unigram_in_vectors
    return vector_multiword

def getting_vectors(vectors, set_words,dimensions):
    # Reads input vectors file and stores the vectors of the words occurring in the dataset in a dictionary
    set_vectors = {}
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        vec = np.array(v)

        if word in set_words:
            set_vectors[word] = vec

    for word in set_words:
        if word not in set_vectors:
            set_vectors[word] = compose_vectors_multiword(word,vectors, dimensions)
    return set_vectors, dimensions
def check(cluster, input_vectors):

    clusternew=set()
    outliernew =set()
    index1,index2=[],[]
    count=0
    for element in cluster.elements:
        if np.sum(input_vectors[element])!=0.0:
            clusternew.add(element)
            index1.append(count)
        count+=1
    count=0
    for element in cluster.outliers:
        if np.sum(input_vectors[element])!=0.0:
            outliernew.add(element)
            index2.append(count)
        count += 1
    cluster.elements=clusternew
    cluster.outliers=outliernew

    if (len(cluster.elements)>2) and (len(cluster.outliers)>0):

        return True,cluster
    else:
        return False, cluster




def outlier(vectors,dim):
    print("Outlier detection test is running....")
    path_dataset = 'wiki-sem/'
    dataset = OutlierDetectionDataset(path_dataset)
    dataset.readDataset()
    input_vectors, dimensions = getting_vectors(vectors, dataset.setWords,dim)
    result,result1,result2 = {},{},{}
    collection=[]
    dictCompactness = {}
    countTotalOutliers,TotalOutliers = 0,0
    numOutliersDetected = 0
    sumPositionsPercentage = 0
    detailedResultsString = ""
    for cluster in dataset.clusters :# 8 different files
        detailedResultsString += "\n\n -- " + cluster.topic + " --\n"
        TotalOutliers+=len(cluster.outliers)
        ischeck,clusternew=check(cluster,input_vectors)
        if (ischeck):
            cluster.elements=clusternew.elements
            cluster.outliers=clusternew.outliers
        else:
            continue

        dictSim = pairwisesimilarities_cluster(cluster.elements, input_vectors)
        numOutliersDetectedCluster = 0
        sumPositionsCluster = 0
        countTotalOutliers += len(cluster.outliers)
        for outlier in cluster.outliers: # one outlier has 8 elements
            compScoreOutlier = 0.0
            dictCompactness.clear()
            for element_cluster_1 in cluster.elements: # 8 same group of elements
               # print(element_cluster_1, outlier)
                sim_outlier_element = cosine(input_vectors[element_cluster_1], input_vectors[outlier]) # cosine similarity between outlier and cluster
                compScoreElement = sim_outlier_element
                compScoreOutlier += sim_outlier_element # summation of cosine of specifc outlier with all cluster elements
                for element_cluster_2 in cluster.elements:
                    if element_cluster_1 != element_cluster_2:
                        compScoreElement += dictSim[element_cluster_1 + " " + element_cluster_2] # sumation of cos(out,e1)+cos(e1,e2)+cos(e1,e3)+...+cos(e1+e8)
                dictCompactness[element_cluster_1] = compScoreElement # sumation of cos(out,e1)+cos(e1,e2)+cos(e1,e3)+...+cos(e1+e8)
                detailedResultsString += "\nP-compactness " + element_cluster_1 + " : " + str(
                    compScoreElement / len(cluster.elements))
            dictCompactness[outlier] = compScoreOutlier #cos(01,e1)+cos(01,e2)...+cos(o1,e8)
            detailedResultsString += "\nP-compactness " + outlier+ " : " + str(
                compScoreOutlier / len(cluster.elements))
            sortedListCompactness = (sorted(iter(dictCompactness.items()), key=operator.itemgetter(1), reverse=True))
            position = 0
            for element_score in sortedListCompactness:
                element = element_score[0]
                if element == outlier:
                    sumPositionsCluster += position
                    if position == len(cluster.elements): numOutliersDetectedCluster += 1
                    break
                position += 1
            detailedResultsString += "\nPosition outlier " + outlier + " : " + str(
                position) + "/" + str(len(cluster.elements)) + "\n"

        numOutliersDetected += numOutliersDetectedCluster
        sumPositionsPercentage += (sumPositionsCluster * 1.0) / len(cluster.elements)
        scoreOPP_Cluster = (((sumPositionsCluster * 1.0) / len(cluster.elements)) / len(cluster.outliers)) * 100
        accuracyCluster = ((numOutliersDetectedCluster * 1.0) / countTotalOutliers) * 100.0
        result[cluster.topic]=(scoreOPP_Cluster,accuracyCluster,len(cluster.outliers))
    scoreOPP = ((sumPositionsPercentage * 1.0) / countTotalOutliers) * 100
    accuracy = ((numOutliersDetected * 1.0) / countTotalOutliers) * 100.0
    result1["Test"] ="Outlier Detection-Accuracy"
    result1["Score"] =accuracy
    result1["OOV"] =str(countTotalOutliers)+"/"+str(TotalOutliers)
    result2["Test"] ="Outlier Detection-OOP"
    result2["Score"] =scoreOPP
    result2["OOV"] =str(countTotalOutliers)+"/"+str(TotalOutliers)
    collection.append(result1)
    collection.append(result2)
    pprint(collection)
    return collection

def pprint(collection):
    print("Outlier detection test result....")
    x = PrettyTable(["Test", "Score", "Not Found/Total"])
    x.align["Dataset"] = "l"
    for result in collection:
        v = []
        for k, m in result.items():
            v.append(m)
        x.add_row([v[0], v[1], v[2]])

    print(x)
    print("-------------------------------------------------------------------------")