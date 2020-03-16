print('Importing..')

import sys
import pandas as pd 
import math
import numpy as np
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from structures import *

##Crawler
print('Crawling..')
#Crawler for 20 newsgroups
import os
 
df = pd.DataFrame(columns=['docID','content','class'])

rootDir = '20_newsgroups'
for dirName, subdirList, fileList in os.walk(rootDir):
    #print('Found directory: %s' % dirName)
    for fname in fileList:
        if fname != '.DS_Store':
            path =  dirName + '/' + fname
            with open(path, 'r', errors='ignore') as myfile:
                data = myfile.read()
                toadd = pd.Series([path,data.lower(),dirName.lower()], index = df.columns)
                df = df.append(toadd, ignore_index=True)

                
#Preprocessing
print('Preprocessing..')
#Re index string to get ride of tags
for i in range(df.shape[0]):
    art = df.iloc[i].content
    ind = art.find('\n\n')
    art = art[ind+2:]
    #If article(and not post), re-index again to get rid of tags
    if art[0:10].find('archive') != -1:
        ind = art.find('\n\n')
        art = art[ind+2:]
    df.iloc[i]['content'] = art 

# Tokenization, stop-word and punctuation removal

from nltk.tokenize import RegexpTokenizer
from nltk import stem
from nltk.corpus import stopwords

ps = stem.PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')

for i in range(df.shape[0]):
    if i % 1000 == 0:
        print('Stemming/tokenizing..')
    removed_punct = tokenizer.tokenize(df.iloc[i].content)
    stems = [ps.stem(word) for word in removed_punct]
    df.iloc[i]['content'] = stems     
    
#create dictionary
d = {}
for i in range(df.shape[0]):
    if i % 1000 == 0:
        print('Indexing..')
    docID = df.iloc[i].docID
    c = pd.DataFrame(df.iloc[i].content, columns=['stems'])
    counts = pd.Series(c.stems.value_counts())
    for term, count in zip(counts.index,counts.values):
        occurence = TokenOccurence(docID,count)
        if term in d.keys():
            occList = d.get(term).append(occurence)
        else:
            d.update({term:[occurence]})
            
print('Consolidating index...')
#Calculate idf
keys = list(d.keys())
for key in keys:
    if key in stopwords.words('english'):
        d.pop(key)
    else: 
        tokenInfo = TokenInfo()
        tokenInfo.occList = d.get(key)
        idf = math.log2(df.shape[0]/len(tokenInfo.occList))
        tokenInfo.idf = idf
        d.update({key:tokenInfo})
        
print('Writing index to disk...')
#save index and classes to disk 
#write postings to disk
import pickle
toFile = open(r'index.pkl', 'wb')
pickle.dump(d, toFile)
toFile.close()

#write classes to disk
classes = df[['docID','class']]
classes.to_csv('classes.csv')
