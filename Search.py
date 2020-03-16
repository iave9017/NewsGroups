
import sys
import pandas as pd
import numpy as np
import math
import pickle 
from structures import TokenInfo, TokenOccurence
from nltk.corpus import stopwords
from nltk import stem
from nltk.corpus import wordnet
import nltk
nltk.download('wordnet')

def getIndex():
    #Write pickle file back to dictionary 
    toDict = open(r'index.pkl', 'rb')
    docTermIndex = pickle.load(toDict)
    toDict.close()
    return docTermIndex


def search(query,docTermIndex):
    print('\nRetrieving documents for query \'{}\'\n'.format(query))
    qlist = query.strip().split()
    #Remove stop words
    modified = [term for term in qlist if term not in stopwords.words('english')]

    EXPAND = False
    '''Expansion
    EXPAND = True
    for term in modified:
        syns = wordnet.synsets(term) 
        for i in range(len(syns)):
            #print(syns[i].lemmas()[0].name())
            new.append(syns[i].lemmas()[0].name())
    '''''''''
    if EXPAND == False:
        new = modified
    before_stem = np.unique(new)
    ps = stem.PorterStemmer()
    after_stem = [ps.stem(word) for word in before_stem]
    mQuery = np.unique(after_stem)
    mQuery

    #read potentially relevant docs into matrix using tf*idf weights 
    qMatrix = pd.DataFrame(np.zeros((0,len(mQuery))), columns = mQuery)

    for term in mQuery:
        if term in docTermIndex.keys():
            termInfo = docTermIndex.get(term)
            for occurence in termInfo.occList:
                if occurence.docID not in qMatrix.index:
                    toAppend = pd.Series(np.zeros(len(qMatrix.columns)), index = qMatrix.columns, name = occurence.docID)
                    toAppend[term] = occurence.count * termInfo.idf
                    qMatrix = qMatrix.append(toAppend)
                else:
                    qMatrix.loc[occurence.docID,term] = occurence.count * termInfo.idf
                    
    #compute tfxidf vector of query 
    #print(qMatrix.columns)
    q_vect = [docTermIndex.get(term).idf for term in qMatrix.columns]

    #Get cosine similarities for query 
    matrix_norm = np.array([np.linalg.norm(qMatrix.iloc[i]) for i in range(len(qMatrix))])
    q_norm = np.linalg.norm(q_vect)
    sims = np.dot(qMatrix,q_vect)/(matrix_norm * q_norm)
    dists = 1 - sims
    idx = np.argsort(dists)
    
    user_docs = qMatrix.iloc[idx[:10]].index 
    classes = pd.read_csv('classes.csv',index_col=1)
    
    for i, path in enumerate(user_docs):
        parts = path.split('/')
        group = parts[1]
        file = parts[2]
        print('----{}: File {} in folder {}-----\n'.format(i+1,file,group))
        with open(path, 'r', errors='ignore') as myfile:
            data = myfile.read()
            art = data
            ind = art.find('\n\n')
            art = art[ind+2:]
            #If article(and not post), re-index again to get rid of tags
            if art[0:10].find('archive') != -1:
                ind = art.find('\n\n')
                art = art[ind+2:]
            mid = len(art)//2
            midmid = mid//2
            print('---------------------------------------------\n')
            print(art[mid:mid+200])
            print('---------------------------------------------\n')
    return user_docs
        