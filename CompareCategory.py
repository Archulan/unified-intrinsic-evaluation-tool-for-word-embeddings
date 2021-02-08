from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tkinter as tk
from tkinter import ttk
from tkinter import *
import json
import numpy as np
from math import sqrt
import re
import logging
import sklearn.decomposition as decomposition
import sklearn.neighbors as neighbors
from matplotlib import pyplot
import io
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Window:
    def __init__(self, app):
        self.lbl1 = Label(app, text='Model_1')
        self.lbl2 = Label(app, text='Model_2')
        self.lbl3 = Label(app, text='Select Category for compare ')
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
        self.category = ttk.Combobox(app,
                                   state="readonly")

        self.category.place(x=400, y=50)
        self.category_dict = {}
        #self.model1.bind("<<ComboboxSelected>>", self.setproperties)
        self.readrepo()
        self.loadCategory()
        self.btn1 = Button(app, text='Compare')
        self.btn1.bind('<Button-1>', self.setproperties)
        self.btn1.place(x=400, y=125)
        self.model1_vectors={}
        self.model2_vectors = {}
        self.model1_dimension=0
        self.model2_dimension = 0
        self.inputcate=""
        self.model1_Name=""
        self.model2_Name = ""
        self.app=app
        app.mainloop()

    def setproperties (self, event):
        print (self.model1.get(),self.model1.current(),self.category.get())
        index1=self.model1.current()
        self.model1_vectors = self.repositories[index1]["vectors"]
        index2 = self.model2.current()
        self.model2_vectors = self.repositories[index2]["vectors"]
        self.model1_dimension = self.repositories[index1]["dimension"]
        self.model2_dimension = self.repositories[index2]["dimension"]
        self.inputcate=self.category.get()
        self.model1_Name=self.model1.get()
        self.model2_Name = self.model2.get()
        if(index1==index2):
            self.error['text'] = "Choose different models to compare"
        else:
            self.app.destroy()
            self.compare(self.inputcate)
        return "break"

    def compare(self,word):
        neighbours=100
        category=self.category_dict[word]

        neighbors_1,words_1= self.get_embedding(self.model1_vectors,neighbours,category,self.model1_dimension)
        neighbors_2, words_2= self.get_embedding(self.model2_vectors, neighbours,category, self.model2_dimension)
        pca_1= self.run_pca(neighbors_1)
        pca_2 = self.run_pca(neighbors_2)
        self.show(pca_1,pca_2,words_1,words_2)

        return

    def get_embedding(self,vectors,neighbours,category,dimension):
        embeddings = np.zeros((neighbours, dimension))
        words=[]
        i=0
        for word, vec in vectors.items():
            if word in category:
                embeddings[i][:]=vec
                words.append(word)
                i+=1
            if i==100:
                break
        return embeddings,words


    def run_pca (self, embeddings):

        pca = decomposition.PCA(n_components=2)
        return pca.fit_transform(embeddings)

    def show (self, embeddings_pca_1,embeddings_pca_2, words1, words2):

        pyplot.figure(figsize=(12, 6))
        pyplot.subplot(121)
        pyplot.gca().set_title(self.model1_Name)

        for i, word in enumerate(words1):
            pyplot.scatter(embeddings_pca_1[i, 0], embeddings_pca_1[i, 1], marker='x', color='blue')
            pyplot.text(embeddings_pca_1[i, 0] + .03, embeddings_pca_1[i, 1] + .03, word, fontsize=9, color='blue')

        pyplot.subplot(122)
        pyplot.gca().set_title(self.model2_Name)

        for i, word in enumerate(words2):
            pyplot.scatter(embeddings_pca_2[i, 0], embeddings_pca_2[i, 1], marker='x', color='blue')
            pyplot.text(embeddings_pca_2[i, 0] + .03, embeddings_pca_2[i, 1] + .03, word, fontsize=9,color='blue')
        pyplot.show()

    def readrepo (self):
        outfile_path = "Repository.json"
        val=[]
        with open(outfile_path) as json_file:
            repository= json.load(json_file)
            self.repositories = repository["Repositories"]

            for repo in self.repositories:
                val.append(repo["Name"])

            self.model1.config(values=val)
            self.model1.current(0)
            self.model2.config(values=val)
            self.model2.current(1)

    def loadCategory(self):
        categories="category.json"
        keys=[]
        with open(categories) as json_file:
            self.category_dict= json.load(json_file)
            for key,val in (self.category_dict.items()):
                keys.append(key)

        self.category.config(values=keys)
        self.category.current(0)

class Factory:
    def __init__(self):
        self.app= tk.Tk()

        self.app.geometry('600x400')

        labelTop = tk.Label(self.app,
                            text="Comparative Evaluation", font='none 16 bold')
        labelTop.config(anchor=CENTER)
        labelTop.pack()
        win=Window(self.app)

    def getapp(self):
        return self.app

    def destroyapp(self):
        self.app.destroy()

