import nltk
import os
import csv
import numpy as np
import tensorflow as tf
import collections

import model



emb_data = "glove.6B.300d.txt"


def build_scalar_encoding(data):
    w_c = collections.Counter(data).most_common() # word/count pairs
    dictionary = dict()
    for word, _ in w_c:
        dictionary[word] = len(dictionary)
    rev_dict = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, rev_dict

def read_data(raw_text):
    content = raw_text
    content = content.split() #splits the text by spaces (default split character)
    content = np.array(content)
    content = np.reshape(content, [-1, ])
    return content


fable_text = """
long ago , the mice had a general council to consider what measures
they could take to outwit their common enemy , the cat . some said
this , and some said that but at last a young mouse got up and said
he had a proposal to make , which he thought would meet the case . 
you will all agree , said he , that our chief danger consists in the
sly and treacherous manner in which the enemy approaches us . now , 
if we could receive some signal of her approach , we could easily
escape from her . i venture , therefore , to propose that a small
bell be procured , and attached by a ribbon round the neck of the cat
. by this means we should always know when she was about , and could
easily retire while she was in the neighbourhood . this proposal met
with general applause , until an old mouse got up and said that is
all very well , but who is to bell the cat ? the mice looked at one
another and nobody spoke . then the old mouse said it is easy to
propose impossible remedies .
"""

def load_embeddings(training_data):
    emb_reader = open(emb_data, 'r', encoding='UTF-8')
    
    vocab = [] # vocab of glove
    embed = [] # embeddings of glove
    emb_dict = {} # dict word-embedding of glove

    #Load embeddings from pretrained model

    for line in emb_reader.readlines():
        elem = line.strip().split(' ')
        wrd = elem[0]
        vec = [float(i) for i in elem[1:]]
        vocab.append(wrd)
        embed.append(vec)
        emb_dict[wrd] = vec
    embedding_dim = len(vec)
    print(embedding_dim)
    print("Loaded Glove")

    #TODO load training data in desirable form

    
    dictionary, rev_dictionary = build_scalar_encoding(training_data) # dictionary word-scalar in trainin data

    #Create embedding array

    tmp_array = []
    dict_size = len(dictionary)
    dict_as_list = sorted(dictionary.items(), key = lambda x : x[1])

    for i in range(dict_size):
        elem = dict_as_list[i][0]
        if elem in vocab:
            tmp_array.append(emb_dict[elem])
        else:
            rand = np.random.uniform(low=-0.2, high=0.2, size=embedding_dim)
            tmp_array.append(rand)

    embedding = np.asarray(tmp_array) # array embedding is set on specific position(word id)
    print(len(embedding))
    print("\n")
    print(embedding[1])
    
    return dictionary, embedding, dict_size, embedding_dim
    

if __name__ == "__main__":
    training_data = read_data(fable_text)
    #dictionary, embedding, dict_size, embedding_dim = load_embeddings(training_data)
    net = model.Model()
    #net.inference([[1, 2, 3, 4, 5], [10, 9, 8, 7, 6]])

    in_words = np.array(["long ago mice general venture", "long ago mice general venture"])
    in_sents = ["Good day.", "Good day."]
    print(in_words, in_sents)
    
    net.inference(in_words, in_sents)
    

