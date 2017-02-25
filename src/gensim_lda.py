#! /usr/bin/env python3

'''
This script runs the topic modeling algorithm (LDA) in the gensim library.
The only dependency is gensim. Install with:

    pip install gensim

If you don't already have it, you may also need to download the stopwords corpus from NLTK corpora.
This is adapted directly from gensim's tutorials:
    https://radimrehurek.com/gensim/tut1.html
    https://radimrehurek.com/gensim/tut2.html
    http://radimrehurek.com/topic_modeling_tutorial/2%20-%20Topic%20Modeling.html
'''

import sys, glob, json
from gensim import corpora, models
from collections import defaultdict
from gensim.parsing.preprocessing import STOPWORDS
import networkx as nx
# from networkx.algorithms import bipartite
from networkx.readwrite import json_graph

sourcedir = sys.argv[1]
num_topics = int(sys.argv[2]) #The number of topics the model will use
files = glob.glob(sourcedir+'*.txt')

# Read files, store in list of strings
data = []
for f in files:
    with open(f, 'r') as newfile:
        newdata = newfile.read()
        data.append(newdata)

#Filter out stopwords (using NLTK stopwords list) and words that appear only once in the corpus, strip punctuation

texts = [[word.strip('.,":;!?()[]\u201d\u201c') for word in document.lower().split() if word not in STOPWORDS] for document in data]


#Create a gensim "dictionary" (not the same as a Python dict) where each word in the corpus is represented by a unique id
dictionary = corpora.Dictionary(texts)

print(dictionary)

#Ignore words that appear in less than 20 documents or more than 20% documents
dictionary.filter_extremes(no_above=0.1)

print(dictionary)
#Convert the corpus into a series of vectors, with words (represented as ids) and their raw counts
corpus = [dictionary.doc2bow(text) for text in texts]

#Fit an LDA model over the corpus
lda = models.LdaModel(corpus, id2word=dictionary, num_topics=num_topics)

#Return topic distribution for all documents (create a new combined corpus using the lda model)
corpus_lda = lda[corpus]

topics = {t[0]:t[1] for t in lda.print_topics(num_topics)}
addresses = [f.split('/')[1].split('.')[0] for f in files]

edges = [[(addresses[idx], t[0], t[1]) for t in y] for idx,y in enumerate(corpus_lda)]
edges = sum(edges, [])

#Print distributions of topics in each document
for idx,y in enumerate(corpus_lda):
    print(addresses[idx], y)

#Print topics
for t in lda.print_topics(num_topics):
    print("Topic "+str(t[0])+": "+t[1])
    print()



#Create bipartite network based on topic modeling data
B = nx.Graph()
B.add_nodes_from(addresses, bipartite=0)
B.add_nodes_from(list(topics.keys()), bipartite=1)
B.add_weighted_edges_from(edges)

nx.set_node_attributes(B, 'topic_words', topics)

# Create a dictionary for the JSON needed by D3.
new_data = json_graph.node_link_data(B) #dict(
#         nodes=[dict(
#             id=n,
#             topic=B.node[n]['topic_words']) for n in B.nodes()],
#         links=[dict(
#             source=e[0],
#             target=e[1],
#             weight=e[2]['weight']) for e in B.edges(data=True)])

# Output json of the graph.
with open('data/inaugural.json', 'w') as output:
        json.dump(new_data, output, sort_keys=True, indent=4, separators=(',',':'))
