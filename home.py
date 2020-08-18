import argparse

# Create the parser
input_parser = argparse.ArgumentParser(prog='Unified intrinsic evaluation tool for word embeddings',
                                       usage='%(prog)s [options] path',
                                       description='Input your model and dimension size')

# Add the arguments
input_parser.add_argument('name',
                          type=str,
                          help='name of the model')

input_parser.add_argument('dim',
                          type=int,
                          help='dimension of the model')

input_parser.add_argument('des',
                          type=str,
                          help='description of the model')

# Execute the parse_args() method
args = input_parser.parse_args()

print(args,'args')
