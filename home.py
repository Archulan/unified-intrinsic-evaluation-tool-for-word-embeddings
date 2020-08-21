import os
import argparse
# from flask import render_template, redirect, url_for, request, flash
# from werkzeug.utils import secure_filename
from distance import similarity
from word_analogy import analogy
from conceptcate import categorize
from OutlierDetection import outlier as out

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

model = args.model
dim = args.dim

result = similarity(model, dim)
result1 = analogy(model, dim)
result2 = categorize(model,dim)
result3 = out(model,dim)
result.extend(result1)
result.append(result2)
result.append(result3)

print(result,'res')

sim, rw, synana, semana, ambi, concept, outlier, oop = 0, 0, 0, 0, 0, 0, 0, 0
for res in result:
    if res["Test"] == "Word similarity":
        sim = res["Score"]
    elif res["Test"] == "RW similarity":
        rw = res["Score"]
    elif res["Test"] == "Syn analogy":
        synana = res["Score"]
    elif res["Test"] == "Sem analogy":
        semana = res["Score"]
    elif res["Test"] == "Concept categorization":
        concept = res["Score"]
    elif res["Test"] == "Outlier Detection":
        outlier = res["Score"]
    elif res["Test"] == "Outlier-oop":
        oop = res["Score"]
    elif res["Test"] == "Ambiguity":
        ambi = res["Score"]
