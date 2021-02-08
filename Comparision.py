from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tkinter as tk
from tkinter import ttk
from tkinter import *
import json
import numpy as np
from math import sqrt
import nltk
from nltk.corpus import wordnet
import sklearn.decomposition as decomposition
import sklearn.neighbors as neighbors
from matplotlib import pyplot
import io
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Window:
    def __init__(self, app):
        self.lbl1 = Label(app, text='Model_1')
        self.lbl2 = Label(app, text='Model_2')
        self.lbl3 = Label(app, text='Input word for analysing its neighbourhood : ')
        self.error = Label(app, text='',fg = "red")
        self.lbl1.place(x=100, y=100)
        self.lbl2.place(x=100, y=150)
        self.lbl3.place(x=100, y=50)
        self.error.place(x=100,y=175)
        self.model1 = ttk.Combobox(app,
                                    state="readonly")

        self.model1.place(x=200, y=100)

        self.model2 = ttk.Combobox(app,
                                   state="readonly")
        self.model2.place(x=200, y=150)
        self.t1 = Entry(bd=3)
        self.t1.place(x=400, y=50)

        self.readrepo()


        self.btn1 = Button(app, text='Compare')
        self.btn1.bind('<Button-1>', self.setproperties)
        self.btn1.place(x=400, y=125)
        self.model1_vectors={}
        self.model2_vectors = {}
        self.model1_dimension=0
        self.model2_dimension = 0
        self.inputword=""
        self.model1_Name=""
        self.model2_Name = ""
        self.app=app
        app.mainloop()

    def setproperties (self, event):
        index1 = self.model1.current()
        index2 = self.model2.current()

        self.model1_vectors = self.repositories[index1]["vectors"]
        self.model2_vectors = self.repositories[index2]["vectors"]
        self.model1_dimension = self.repositories[index1]["dimension"]
        self.model2_dimension = self.repositories[index2]["dimension"]
        self.model1_Name = self.model1.get()
        self.model2_Name = self.model2.get()
        self.inputword = self.t1.get()
        if index1 == index2:
            self.error['text'] = "Choose different models to compare"
        elif self.inputword == "":
            self.error['text'] = "Please type Input word for analysing its neighbourhood"
        elif self.inputword not in self.model1_vectors.keys() or self.inputword not in self.model2_vectors.keys():
            self.error['text'] = "Input word is not exist in the vocabulary of the models"
        else:
            self.app.destroy()
            self.compare(self.inputword)
        return "break"

    def compare(self, word):
        neighbours = 25
        vec1 = self.model1_vectors[word]
        vec2 = self.model2_vectors[word]
        neighbors_1 = self.get_neighbors(self.model1_vectors,vec1,neighbours)
        neighbors_2 = self.get_neighbors(self.model2_vectors, vec2, neighbours)
        pca_1, words_1 = self.run_pca(neighbors_1,self.model1_vectors,self.model1_dimension,neighbours)
        pca_2, words_2 = self.run_pca(neighbors_2, self.model2_vectors, self.model2_dimension,neighbours)
        self.show(pca_1,pca_2,words_1,words_2,word)

        return

    def euclidean_distance(self,row1, row2, dim):
        distance = 0.0
        for i in range(dim):
            distance += (row1[i] - row2[i]) ** 2
        return sqrt(distance)

    def get_neighbors(self,vectors, test_row, num_neighbors):
        distances = list()
        count = 0
        dim = len(test_row)
        for word, train_row in vectors.items():
            dist = self.euclidean_distance(test_row, train_row, dim)
            distances.append((count, dist))
            count += 1
        distances.sort(key=lambda tup: tup[1])
        neighbors = []

        for i in range(num_neighbors):
            neighbors.append(distances[i][0])
        return neighbors

    def run_pca (self, neighbors, vectors, dimension,neighbours_num):
        index,i = 0,0
        embeddings = np.zeros((neighbours_num, dimension))
        words=[]
        for word, vector in vectors.items():
            if index in neighbors:
                embeddings[i][:]=vector
                words.append(word)
                i+=1
            index+=1

        pca = decomposition.PCA(n_components=2)
        return pca.fit_transform(embeddings) , words

    def show (self, embeddings_pca_1,embeddings_pca_2, words1, words2,input):

        pyplot.figure(figsize=(12, 6))
        ax=pyplot.subplot(221)
        ax.set_title(self.model1_Name)

        for i, word in enumerate(words1):
            if word==input:
                pyplot.scatter(embeddings_pca_1[i, 0], embeddings_pca_1[i, 1], marker='x', color='black')
                pyplot.text(embeddings_pca_1[i, 0] + .03, embeddings_pca_1[i, 1] + .03, word, fontsize=10, color='black')
            elif word in words2:
                pyplot.scatter(embeddings_pca_1[i, 0], embeddings_pca_1[i, 1], marker='x', color='red')
                pyplot.text(embeddings_pca_1[i, 0] + .03, embeddings_pca_1[i, 1] + .03, word, fontsize=9,color='red')
            else:
                pyplot.scatter(embeddings_pca_1[i, 0], embeddings_pca_1[i, 1], marker='x', color='blue')
                pyplot.text(embeddings_pca_1[i, 0] + .03, embeddings_pca_1[i, 1] + .03, word, fontsize=9, color='blue')

        ax=pyplot.subplot(222)
        ax.set_title(self.model2_Name)
        for i, word in enumerate(words2):
            if word==input:
                pyplot.scatter(embeddings_pca_2[i, 0], embeddings_pca_2[i, 1], marker='x', color='black')
                pyplot.text(embeddings_pca_2[i, 0] + .03, embeddings_pca_2[i, 1] + .03, word, fontsize=10, color='black')
            elif word in words1:
                pyplot.scatter(embeddings_pca_2[i, 0], embeddings_pca_2[i, 1], marker='x', color='red')
                pyplot.text(embeddings_pca_2[i, 0]+.03, embeddings_pca_2[i, 1]+.03, word, fontsize=9,color='red')
            else:
                pyplot.scatter(embeddings_pca_2[i, 0], embeddings_pca_2[i, 1], marker='x', color='blue')
                pyplot.text(embeddings_pca_2[i, 0] + .03, embeddings_pca_2[i, 1] + .03, word, fontsize=9,color='blue')

        pyplot.subplot(223)
        pyplot.text(0.25,1," Synsets from wordnet")

        pyplot.axis(False)
        text=self.get_synsets(input)
        figtext_args = (0.2, 0.35,
                        text)

        figtext_kwargs = dict(horizontalalignment="left",
                              fontsize=10, color="black",
                               wrap=True)


        pyplot.figtext(*figtext_args, **figtext_kwargs)
        #print(text)

        pyplot.show()
    def get_synsets(self,word):
        result=""
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                if l.name() != word:
                    result+=(l.name()+"  ")
        return result


    def readrepo (self):
        outfile_path = "Repository.json"
        val=[]
        with open(outfile_path) as json_file:
            repository= json.load(json_file)
            self.repositories = repository["Repositories"]
            #print ("repositories: ",len(self.repositories))
            for repo in self.repositories:
                val.append(repo["Name"])

            self.model1.config(values=val)
            self.model1.current(0)
            self.model2.config(values=val)
            self.model2.current(1)

class Factory:
    def __init__(self):
        self.app= tk.Tk()

        self.app.geometry('600x400')

        labelTop = tk.Label(self.app,
                            text="Comparative Evaluation", font='none 16 bold')
        labelTop.config(anchor=CENTER)
        labelTop.pack()
        win=Window(self.app)

