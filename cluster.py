from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import numpy as np
import sklearn.decomposition as decomposition
import sklearn.neighbors as neighbors
from matplotlib import pyplot

# Round all floats in JSON dump to 5 decimal places.
json.encoder.FLOAT_REPR = lambda x: format(x, '.5f')

METRICS = ['cosine', 'euclidean']

#FLAGS = flags.FLAGS

#flags.DEFINE_integer(
 #   'max_k',
#250,
#    'Max value of K for defining local neighborhoods (default = 250)')
#flags.DEFINE_string('embeddings_file', None, 'Path to embeddings file (tsv).')
#flags.DEFINE_string('metadata_file', None, 'Path to metadata file (tsv).')
#flags.DEFINE_string('outfile', None, 'Path to write preprocessed data (json).')


def load_embeddings(filepath):
    feed = open(filepath,'r', encoding="utf8").read()
    num_lines = len(feed.splitlines())
    dimension=len(feed.splitlines()[0].split(" ")) -1
    embeddings=np.zeros((num_lines,dimension))
    print(embeddings.shape)
    with open(filepath, 'r', encoding="utf8") as f:
        count=0
        for row in f:
            word=row.strip().split(" ")[0]
            if word.isalpha():
                line=row.strip().split(" ")[1:]
                list1=np.array([float(x) for x in line])
                embeddings[count][:]=list1
                count+=1
    print(count, num_lines)
    embeddings=embeddings[:count]
    print(embeddings.shape)
    return embeddings


def load_words(filepath):
    words = []
    with open(filepath, 'r', encoding="utf8") as f:
        for row in f:
            word = row.strip().split(" ")[0]
            if word.isalpha():
                words.append(word)
    print(len(words))
    return words


def compute_nearest_neighbors(embeddings, max_k, metric):
    print("Compute nearest neighbours")
    neigh = neighbors.NearestNeighbors(n_neighbors=max_k, metric=metric,n_jobs=500)
    neigh.fit(embeddings)
    dist, ind = neigh.kneighbors(return_distance=True)
    return ind, dist


def create_nearest_neighbors_dicts(embeddings, max_k, metrics):
    to_return = [
        {metric: None for metric in metrics} for _ in range(len(embeddings))
    ]
    for metric in metrics:
        inds, dists = compute_nearest_neighbors(embeddings, max_k, metric)
        for i, (ind, dist) in enumerate(zip(inds, dists)):
            to_return[i][metric] = {
                'knn_ind': ind.tolist(),
                'knn_dist': dist.tolist(),
            }
    return to_return


def create_preprocessed_data(embeddings, words, nn_dicts, embeddings_pca):
    print("create preprocessed data")
    to_return = []
    for i, (embedding, word, nn_dict, embedding_pca) in enumerate(
        zip(embeddings, words, nn_dicts, embeddings_pca)):
        to_return.append({
            'idx': i,
            'word': word.lower(),
            'embedding': list(embedding),
            'nearest_neighbors': nn_dict,
            'embedding_pca': list(embedding_pca),
        })
    return to_return


def run_pca(embeddings):
    print("run pca")
    pca = decomposition.PCA(n_components=2)
    return pca.fit_transform(embeddings)


def write_outfile(outfile_path, preprocessed_data):
    print("write file")
    with open(outfile_path, 'w', encoding="utf8") as f:
        json.dump(preprocessed_data, f, separators=(',', ':'))

def show(words,embeddings_pca,words2, embeddings_pca2):
    print("show")
    #pyplot.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1])
    #for i, word in enumerate(words):
     #   if word in words2:
      #      pyplot.annotate(word, xy=(embeddings_pca[i, 0],embeddings_pca[i, 1]),c="r")
    #    else:
    #    pyplot.show()
    pyplot.scatter(embeddings_pca2[:, 0], embeddings_pca2[:, 1])
    for i, word2 in enumerate(words2):
        if word2 in words:
            pyplot.annotate(word2, xy=(embeddings_pca[i, 0], embeddings_pca[i, 1]), c="r")
        else:
            pyplot.annotate(word2, xy=(embeddings_pca[i, 0], embeddings_pca[i, 1]), c="b")
    pyplot.show()
def main():
    #del argv

    #logging.basicConfig(level=logging.INFO)
    #/content/gdrive/My Drive/fyp/eval/python/
    embeddings_file= "/Downloads/GloVe-master/GloVe-master/eval/python/sample_glove.txt"
    outfile_path = "outfile.txt"
    #embeddings_file = path2
    #outfile_path2 = "outfile2.json"
    max_k = 100

    # Load embeddings and words from file.
    embeddings = load_embeddings(embeddings_file)
    words = load_words(embeddings_file)
   # print(embeddings)
    # Compute nearest neighbors.
    nn_dicts = create_nearest_neighbors_dicts(embeddings, max_k, METRICS)

    embeddings_pca = run_pca(embeddings)
    preprocessed_data = create_preprocessed_data(
        embeddings, words, nn_dicts, embeddings_pca)
    write_outfile(outfile_path, preprocessed_data)
    # Load embeddings and words from file.



if __name__ == '__main__':
    #flags.mark_flag_as_required('embeddings_file')
    #flags.mark_flag_as_required('metadata_file')
    #flags.mark_flag_as_required('outfile')
    main()
