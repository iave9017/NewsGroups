import sys
import os
from Search import *
import pickle

docTermIndex = None
print('Importing index for use...If first run, have to create index from documents')
try:
    #index = getIndex()
    toDict = open(r'index.pkl', 'rb')
    docTermIndex = pickle.load(toDict)
    toDict.close()
    # Do something with the file
except IOError:
    print("Index not created yet.. Creating index")
    os.system('python crawl_index_process.py')

print('\n\nReady to query database!\n')
while True:
    print('Type query followed by query to search database')
    print('Type quit to quit program\n')
    myInput = input("Search here>")
    if myInput == 'quit':
        sys.exit(0)
    else:
        flag = myInput.strip().split()
        if flag[0] == 'query':
            top10 = search(myInput[6:].lower(),docTermIndex)
        else:
            print('ERROR: Invalid query. Please try again\n')
            




    


