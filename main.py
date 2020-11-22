import pandas as pd
import os
import sys
import nltk
import xml.etree.ElementTree as ET
import numpy as np

from nltk.tokenize import casual_tokenize
from nltk.stem.snowball import SnowballStemmer
from subprocess import Popen

df = pd.read_csv('pcgarage.csv', delimiter = '\t', encoding = 'utf-16')

#print(df)

df_corpus = df.iloc[:, -3:]

#print(df_corpus)

df_corpus["corpus"] = df_corpus["pro"].astype(str) + " " + \
                        df_corpus["contra"].astype(str) + " " + \
                        df_corpus["altele"].astype(str)
#task1
all_tokens_list = [casual_tokenize(text.lower()) for text in df_corpus["corpus"]]
tokens_list = [[x for x in y if x.isalpha()] for y in all_tokens_list]
tokens = [x for y in tokens_list for x in y]
print(tokens[0:10])


lemmatizer = SnowballStemmer(language='romanian')
lemma = [[lemmatizer.stem(token) for token in y] for y in tokens_list]
print(lemma[0:10])

data = df_corpus["corpus"]

'''i = 1
for rows in np.array_split(data, 10):
    rows.to_csv("inputuri/pc" + str(i) + ".csv", encoding = "UTF-8",  index = False)
    i = i + 1
    if i == 8:
        i = i + 1'''



'''
PATH_TO_BAT = r"C:\Users\Mihai\Desktop\labPython"
BAT_FNAME = "start tagging inputs.bat"
p = Popen(BAT_FNAME , cwd=PATH_TO_BAT)
stdout, stderr = p.communicate()'''

#task 2,3
lemmas = list()
pos_list = list()
for i in range(10):
    try:
        tree = ET.parse('outputuri\pc'+str(i+1)+'.xml')
        root = tree.getroot()
        print(str(i+1))
        for word in root.iter('W'):
            lema = word.get('LEMMA')
            pos = word.get('POS')
            lemmas.append(lema)
            pos_list.append(pos)
    except:
        continue

#task4
stopwords = [',', '.', '/', '\\', ';', ':', "\""]

def wordListToFreqDict(wordlist):
    wordfreq = [wordlist.count(p) for p in wordlist]
    return dict(list(zip(wordlist,wordfreq)))

def sortFreqDict(freqdict):
    aux = [(freqdict[key], key) for key in freqdict]
    aux.sort()
    aux.reverse()
    return aux

def removeStopwords(wordlist, stopwords):
    return [w for w in wordlist if w not in stopwords]


wordlist = removeStopwords(lemmas, stopwords)
dictionary = wordListToFreqDict(wordlist)
sorteddict = sortFreqDict(dictionary)
i = 0
for s in sorteddict:
    print(str(s))
    i = i + 1
    if i == 20:
        break
    else:
        continue
#task5
wordlist = removeStopwords(pos_list, stopwords)
dictionary = wordListToFreqDict(wordlist)
sorteddict = sortFreqDict(dictionary)

i = 0
for s in sorteddict:
    print(str(s))
    i = i + 1
    if i == 20:
        break
    else:
        continue
