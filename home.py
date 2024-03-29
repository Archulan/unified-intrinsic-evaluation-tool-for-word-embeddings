import os
import io
import argparse
import numpy as np
from Main import App
from tqdm import tqdm
from prettytable import PrettyTable
from distance import SimilarityEvaluator
from wordAnalogy import AnalogyEvaluator
from outlierDetection import OutlierEvaluator
from conceptCategorization import CategorizationEvaluator
from Comparision import Factory
from CompareCategory import Factory as Fac
import json
# Create the parser
input_parser = argparse.ArgumentParser(prog='Unified intrinsic evaluation tool for word embeddings',
                                       usage='%(prog)s [options] path',
                                       description='Input your model and dimension size')
input_parser.version = '1.0'


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return open(arg, 'r')


def generate(filename, dim):
    print('Reading model file....')
    words = []
    with tqdm(io.open(filename, 'r', encoding="utf8")) as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            if (vals[0].isalpha() and len(vals) == int(dim) + 1):
                vectors[vals[0]] = [float(x) for x in vals[1:]]
                words.append(vals[0])

    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    W_norm = np.zeros((vocab_size, vector_dim))
    print('Vocabulary size:', vocab_size)

    for word, v in vectors.items():
        if word == '<unk>':
            continue
        vec = np.array(v)
        d = (np.sum((vec) ** 2, ) ** (0.5))
        norm = (vec.T / d).T
        W_norm[vocab[word], :] = norm

    return (W_norm, vocab, ivocab, words, vectors)


def pprint(collections):
    print("Final results")
    x = PrettyTable(["Test", "Score (rho)", "Not Found/Total"])
    x.align["Dataset"] = "l"
    for result in collections:
        v = []
        for k, m in result.items():
            v.append(m)
        x.add_row([v[0], v[1], v[2]])

    print(x)
    print("---------------------------------------")



def show_option_for_compare():
    outfile_path = "Repository.json"
    with open(outfile_path) as json_file:
        repository = json.load(json_file)
        repositories = repository["Repositories"]
    print("Repos: ",len(repositories))
    if len(repositories) >=2:
        while (True):
            option=input("If you want to continue with comparative evaluation \n Press 1 else press 2:")
            if int(option) == 1:
                opt2=input("you have 2 option to do comparative evalution \n Press 1 for compare models neighbourhood using given word \n"
                           "Press 2 for compare the models by choosing set of category words \n")

                if int(opt2) == 1:
                    win = Factory()
                elif int(opt2) == 2:
                    win = Fac()
                else:
                    exit=input("Wrong input \n To exit press -1")
                    if (int(exit) == -1):
                        break
            elif int(option) == 2:
                break
            else :
                print("Wrong input")

# arguments
input_parser.add_argument('name',
                          type=str,
                          help='name of the model')

input_parser.add_argument('model',
                          help="input the model file", metavar="FILE",
                          type=lambda x: is_valid_file(input_parser, x))

input_parser.add_argument('dim',
                          type=int,
                          help='dimension of the model')

input_parser.add_argument('des',
                          type=str,
                          help='description about the model')

input_parser.add_argument('-v',
                          help='Tool version',
                          action='version')

# Execute the parse_args() method
args = input_parser.parse_args()
model = args.model.name
dim = args.dim
name=args.name
print('Name       :', args.name)
print('Model file :', args.model.name)
print('Dimension  :', args.dim)
print('Description:', args.des, '\n')
#show_option_for_compare()
app = App(name=name,dim=dim, path=model,
          plugins=[SimilarityEvaluator(), AnalogyEvaluator(), CategorizationEvaluator(), OutlierEvaluator()])
app.run()
show_option_for_compare()