import io
from tqdm import tqdm
from prettytable import PrettyTable
import json
class App:
    """First attempt at a plugin system"""
    def __init__(self, *,name,dim,path, plugins: list=list()):
        self.name=name
        self.dimension=dim
        self.vec_path=path
        self._plugins = plugins
        self.result=[]

    def generate_vectors(self):
        print('Reading model file....')
        with tqdm(io.open(self.vec_path, 'r', encoding="ISO-8859-1")) as f:
            vectors = {}
            for line in f:
                vals = line.rstrip().split(' ')
                if (vals[0].isalpha() and len(vals) == int(self.dimension) + 1):
                    vectors[vals[0]] = [float(x) for x in vals[1:]]
                    #words.append(vals[0])
        return vectors

    def pprint(self,collections):
        print("Final results")
        x = PrettyTable(["Test", "Score (rho)", "Not Found/Total"])
        x.align["Dataset"] = "l"
        summary=self.generate_summary(collections)
        for result in collections:
            v = []
            for k, m in result.items():
                v.append(m)
            x.add_row([v[0], v[1], v[2]])

        print(x)
        print("------------------------------------------------------")
        print(summary)
        print("------------------------------------------------------")
    def run(self):
        vectors=self.generate_vectors()
        self.addIntoRepo(vectors)
        modules_to_execute =  self._plugins
        for module in modules_to_execute:
            out=module.process(vectors)
            self.result.extend(out)
        self.pprint(self.result)


    def generate_summary(self,result):
        summary = ""
        for test in result:
            type=test["Test"]
            score=int(test["Score"])
            if (type=="Multifactedness"):
                benchmarks=test["Expand"]
                if (score>=80):
                    summary+="Model is very good at to identify morphological semantic and relatedness lexical properties. "
                elif (score<80 and score >=40):
                    relateness=0
                    semantics=0
                    for file in benchmarks:
                        if(file["Test"]=="EN-MEN-TR-3k" and file["Spearman"] >=80):
                            summary += "Model is very good at to identify semantic and relatedness lexical properties. "
                        elif (file["Test"] == "EN-MEN-TR-3k" and file["Spearman"] >= 60):
                            summary += "Model is good at to identify semantic and relatedness lexical properties. "
                        elif (file["Test"]=="EN-WS-353-REL" ):
                            relateness = file["Spearman"]
                            if ( file["Spearman"] >=80):
                                summary+="model is very good at identify relatedness property. "
                            elif (file["spearman"] < 80 and file["Spearman"] >60):
                                summary += "model is good at identify relatedness property. "
                            elif (file["spearman"] < 60 and file["Spearman"] >40):
                                summary += "model shows average performance on identifying relatedness property. "
                            else:
                                summary+= "model shows poor performance in identifying relatedness property. "
                        elif (file["Test"]=="EN-WS-353-SIM" ):
                            semantics = file["Spearman"]
                            if ( file["Spearman"] >=80):
                                summary+="model is very good at identify semantics of the words. "
                            elif (file["spearman"] < 80 and file["Spearman"] >60):
                                summary += "model is good at identify semantics of the words. "
                            elif (file["spearman"] < 60 and file["Spearman"] >40):
                                summary += "model shows average performance on identifying semantics of the words. "
                            else:
                                summary+= "model shows poor performance in identifying semantics of the words. "
                        else:
                            if ( file["Spearman"] >=80):
                                summary+="Model is very good at capturing semantic meaning of the verbs. "
                            elif (file["spearman"] < 80 and file["Spearman"] >60):
                                summary += "model is good at capturing semantic meaning of the verbs. "
                            elif (file["spearman"] < 60 and file["Spearman"] >40):
                                summary += "model shows average performance on capturing semantic meaning of the verbs. "
                            else:
                                summary+= "model shows poor performance on capturing semantic meaning of the verbs. "
                    if (relateness > semantics):
                        summary+= "model performs better in relatedness over semantic meaning. "
                    else:
                        summary+= "model performs better in capturing semantic meaning than relatedness property. "
                else:
                    summary+="Model shows poor performance on identifying morphological semantic and relatedness lexical properties. "
            elif(type=="Sparsness"):
                if (score >80):
                    summary += "Model is very good at capturing rare words. "
                elif (score >60 and score<80):
                    summary += "Model is good at capturing rare words. "
                elif (score > 40 and score < 60):
                    summary += "Model shows avere performance on capturing rare words. "
                else:
                    summary +="Model shows poor perfromance on capturing rare words. try to increase the corpus which used to train and implement subwording technique which enhance the performance of the model "
            elif (type == "Lexical Ambiguity"):
                if (score > 80):
                    summary += "Model is very good at perceiving different meaning of the words. "
                elif (score > 60 and score < 80):
                    summary += "Model is good at perceiving different meaning of the words. "
                elif (score > 40 and score < 60):
                    summary += "Model shows average performance on perceiving different meaning of the words. "
                else:
                    summary += "Model shows poor performance on perceiving different meaning of the words. "
            elif (type == "Non Conflation"):
                if (score > 80):
                    summary += "Model is very good at distinguishing the detailed meaningful aspects of the words. "
                elif (score > 60 and score < 80):
                    summary += "Model is good at distinguishing the detailed meaningful aspects of the words. "
                elif (score > 40 and score < 60):
                    summary += "Model shows average performance on distinguishing the detailed meaningful aspects of the words. "
                else:
                    summary += "Model shows poor performance on distinguishing the detailed meaningful aspects of the words. "

            elif (type == "Handling Hubness"):
                if (score > 80):
                    summary += "Model is very good at Handling Hubness. "
                elif (score > 60 and score < 80):
                    summary += "Model is good at Handling Hubness. "
                elif (score > 40 and score < 60):
                    summary += "Model shows average performance on Handling Hubness. "
                else:
                    summary += "Model shows poor performance on Handling Hubness. Removing very frequent words in the corpus is helped to improve this property "

            elif (type == "Geometry"):
                if (score > 80):
                    summary += "Model is spreaded vectors widely. Frequent words and un related words are placed very correctly "
                elif (score > 60 and score < 80):
                    summary += "Frequent words and un related words are placed correctly "
                elif (score > 40 and score < 60):
                    summary += "Model shows average performance on placing Frequent words and un related words on a correct place. "
                else:
                    summary += "Model is not spreaded vectors widely. Frequent words and un related words are not placed correctly "
            else:
                if (score > 80):
                    summary += "Model have coherent neighbourhood for each word vector. "
                elif (score > 60 and score < 80):
                    summary += "Model have coherent neighbourhood for almost each word vector. "
                elif (score > 40 and score < 60):
                    summary += "Model is little failed to having coherant neighbourhood for each word vector. "
                else:
                    summary += "Model is failed to having coherant neighbourhood for each word vector. "
        return summary

    def addIntoRepo(self,vectors):
        data={}
        data["Name"]=self.name
        data["dimension"]=self.dimension
        data["vectors"]=vectors
        #data["Words"]= words
        outfile_path="Repository.json"
        with open(outfile_path) as json_file:
            repo = json.load(json_file)
            temp=repo["Repositories"]
            temp.append(data)
        with open(outfile_path, 'w') as f:
            json.dump(repo, f, indent=4)
