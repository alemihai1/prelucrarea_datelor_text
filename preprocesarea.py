import pandas as pd
import os
import sys
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
import xml.etree.ElementTree as ET
import numpy as np
import nltk
import string
from nltk.stem import PorterStemmer, SnowballStemmer

from cube.api import Cube
cube=Cube(verbose=True)         
cube.load("ro")                

#import baza
df = pd.read_csv('pcgarage.csv', delimiter = '\t', encoding = 'utf-16', header = 0)

#preprocesare   
#eliminarea elementelor nedorite
df['pro'] = df['pro'].str[4:]
df['contra'] = df['contra'].str[8:]
df['altele'] = df['altele'].str[8:]
#concatenarea partilor
df["corpus"] = df["pro"].astype(str) + " " + df["contra"].astype(str) + " " + df["altele"].astype(str)
data = df[['product', 'rating', 'corpus']].copy()
data['corpus'] = [it.lower().replace('\n\n', ' ') for it in data['corpus']]
#transformarea majusculelor
data['corpus'] = data.corpus.map(lambda x: x.lower())
#tokenizarea
data['corpus'] = data['corpus'].apply(nltk.word_tokenize)
data['corpus'] = data.corpus.map(lambda txt: [word for word in txt if word.isalpha()])
STOPWORDS = set(stopwords.words('romanian'))
PUNCTUATION_SIGNS = [c for c in string.punctuation] + ['``', "''", '...', '..']
data['corpus'] = data.corpus.map(lambda txt: [word.strip() for word in txt if word not in STOPWORDS])
data['corpus'] = data.corpus.map(lambda txt: [word.strip() for word in txt if word not in PUNCTUATION_SIGNS ])

for i in data.index:
    if len(data.loc[i,'corpus']) < 4:
        data = data.drop(i)

stemmer = SnowballStemmer(language="romanian")  # PorterStemmer()
data.head(10)
data['corpus'] = data['corpus'].apply(lambda x: [stemmer.stem(y.strip()) for y in x if y])
data.head(10)

data2 = data[['product', 'rating', 'corpus']].copy()
data1 = list()
for row in data2.iloc[:,2]:
    lemmas = ""
    for word in row:
        sentences = cube(word)
        for entry in sentences[0]:
            lemmas += entry.lemma + " "
            # now, we look for a space after the lemma to add it as well
            if not "SpaceAfter=No" in entry.space_after:
                lemmas += " "
    data1.append(lemmas)
data['text'] = data1

for ind in data.index:
    data['initial'][ind] = df.loc[ind,'corpus']

w = 0
for i in data.index:
    w = w + len(data.loc[i,'text'])
w

data.to_csv('pctoken.csv', sep = '\t', encoding = 'utf-16', index = False, header = False)