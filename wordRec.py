from sklearn.metrics.pairwise import cosine_similarity
from scipy import linalg, mat, dot
import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
import pickle
from gensim.models import Word2Vec
import spacy
nlp = spacy.load('en_core_web_sm')
import flask
import pandas as pd
import math
from flask_cors import CORS
app = flask.Flask(__name__)
CORS(app)
model = Word2Vec.load('./w2v5')
k_means = pickle.load(open('./spectral', 'rb'))
centroids = k_means.cluster_centers_
labels = k_means.labels_
fileB = open('entPickle.p', 'rb')
newEnts = pd.DataFrame(columns = ['entities', 'vectors'])
while 1:
    try:
        newEnts = newEnts.append(pickle.load(fileB), ignore_index = True)
    except EOFError:
        break

@app.route("/recWord", methods = ["POST"])

def recWord():
    data = {"success": False}
    body = flask.request.get_json()
    if flask.request.method == "POST" and len(body["input"]) > 0:
        try: 
            inp = body["input"]
            pdist = []
            cluster = []
            returnedCluster = []
            similarityIndex = []
            res = []
            vec = nlp(unicode(inp)).vector
            inp = inp.split()
            closestList = []
            for ind in centroids:
                arrayCos = []
                indSplit = np.split(ind, 7)
                for x in range(7):
                    arrayCos = np.append(arrayCos, math.sqrt(sum(pow(a-b,2) for a, b in zip(indSplit[x], vec))))
                pdist.append(arrayCos[np.argmin(arrayCos)])
            pdist = np.array(pdist).argsort()
            for x in range(4):
                closestList = np.append(closestList, pdist[x])
            for y in range(len(closestList)):
                for i, x in enumerate(labels):
                    if x == y:
                        try:
                            for x in newEnts['entities'][i]:
                                if x != 0:
                                    cluster.append(x)
                                else:
                                    break
                        except:
                            continue 
            for x in cluster:
                similarity = 0
                clusterWord = [word for word in str(x).split()]
                for x in clusterWord:
                    for y in inp:
                        try:
                            similarity += model.wv.similarity(x.lower(), y.lower())
                        except:
                            pass
                similarityIndex.append(similarity)
            for i, sim in enumerate(similarityIndex):
                if sim > 0.45:
                    returnedCluster.append(cluster[i])
            returnedCluster = str(np.unique(np.array(returnedCluster)))
            data['res'] = returnedCluster
            data['success'] = True
        except IOError:
            print(IOError)
            data['error'] = 'input not in training data...try similar word'
    else:
        data['error'] = 'no input provided'
    return flask.jsonify(data)

if __name__ == "__main__":
    app.debug = True
    print("* Loading Flask Application")
app.run(port = 5051, host = '0.0.0.0')
