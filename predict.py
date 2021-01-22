import tflearn
import string
import pickle
import argparse
import numpy as nm
import tensorflow as tf

def convertTextToIndex(dictionary, text):
    document = []
    text = text.lower().encode('utf-8')
    words = text.split()
    for word in words:
        word = word.translate(None, string.punctuation.encode('utf-8'))
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
        document.append(index)

    ln = 150 - len(document)
    if ln>0 :
        document = nm.pad(document, (0, ln), 'constant')
    return document

def sentiment_init(lang):
    
    net = tflearn.input_data([None, 150])
    net = tflearn.embedding(net, input_dim=10000, output_dim=128)
    net = tflearn.lstm(net, 128, dropout=0.8)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                                 loss='categorical_crossentropy')
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.load("checkpoints/"+lang+"/"+lang+"tf.tfl")
    return model

