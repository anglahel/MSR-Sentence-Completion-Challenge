import nltk
import os
import csv
import numpy as np
import tensorflow as tf
import collections
import pandas as pd
import re
import random
import model
import trainer

def load_queries(dataset):

    df = pd.read_csv(dataset)
    val = df.values
    n = len(val)
    queries = []
    answers = []

    for i in range(n):
        entry = val[i]
        queries.append(entry[1])
        answers.append(entry[2:7])

    return queries,answers



def load_labels(dataset):

    df = pd.read_csv(dataset)
    val = df.values
    n = len(val)
    labels = []

    for i in range(n):
         if(val[i][1]=='a'):
            labels.append(1)
         elif(val[i][1]=='b'):
            labels.append(2)
         elif(val[i][1]=='c'):
            labels.append(3)
         elif(val[i][1]=='d'):
            labels.append(4)
         else:
            labels.append(5)

    return labels



if __name__ == "__main__":

    current_dataset = "training_data.csv"
    current_labels = "train_answer.csv"
    batch_size = 1024
    shuf = True
    fst = True

    sentences, answers = load_queries(current_dataset)
    labels = load_labels(current_labels)


    test_sent, test_answ = load_queries("testing_data.csv")
    test_labels = load_labels("test_answer.csv")

    print("Loaded data\n")

    n = len(sentences)

    if(shuf):

        mall = []

        for i in range(n):
            mall.append([sentences[i],answers[i],labels[i]])

        random.shuffle(mall)

        sentences = []
        answers = []
        labels = []

        for i in range(n):
            sentences.append(mall[i][0])
            answers.append(mall[i][1])
            labels.append(mall[i][2])

        ind = 0
        j = 0



    if(fst):

        mall = []
        for i in range(n):

            tmp = answers[i][labels[i]-1]
            answers[i][labels[i]-1] = answers[i][0]
            answers[i][0] = tmp

            mall.append(str(answers[i][0])+" "+str(answers[i][1])+" "+str(answers[i][2])+" "+str(answers[i][3])+" "+str(answers[i][4]))

        answers = []

        for i in range(n):
            answers.append(mall[i])

    queries = []
    for i in range(n):
        queries.append([sentences[i],answers[i]])

    t_size = 550000
    v_size = 200000

    train_data = queries[0:t_size]
    valid_data = queries[t_size:]
    
    m = len(test_sent)

    test_data = []
    for i in range(m):
        test_data.append([test_sent[i],str(test_answ[i][0])+" "+str(test_answ[i][1])+" "+str(test_answ[i][2])+" "+str(test_answ[i][3])+" "+str(test_answ[i][4]),test_labels[i]])

    model = model.Model()
    trainer = trainer.Trainer(train_data = np.array(train_data),valid_data = np.array(valid_data), test_data=np.array(test_data), epochs = 100, batch_size = batch_size, model = model)

    trainer.train()


