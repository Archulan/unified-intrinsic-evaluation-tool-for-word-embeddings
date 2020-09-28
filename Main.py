import io
import numpy as np
from tqdm import tqdm
from prettytable import PrettyTable

class App:
    """First attempt at a plugin system"""
    def __init__(self, *,dim,path, plugins: list=list()):
        self.dimension=dim
        self.vec_path=path
        self._plugins = plugins
        self.result=[]

    def generate_vectors(self):
        print('Reading model file....')
        with tqdm(io.open(self.vec_path, 'r', encoding="utf8")) as f:
            vectors = {}
            for line in f:
                vals = line.rstrip().split(' ')
                if (vals[0].isalpha() and len(vals) == int(self.dimension) + 1):
                    vectors[vals[0]] = [float(x) for x in vals[1:]]
        return vectors

    def pprint(self,collections):
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

    def run(self):
        vectors=self.generate_vectors()
        modules_to_execute =  self._plugins
        for module in modules_to_execute:
            out=module.process(vectors)
            self.result.extend(out)
        self.pprint(self.result)