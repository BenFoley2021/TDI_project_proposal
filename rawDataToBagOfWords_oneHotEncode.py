# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 16:53:58 2021


@author: bcyk5
"""

import pandas as pd
import spacy
import numpy as np
import time
import scipy.sparse
from scipy.sparse import coo_matrix, lil_matrix

      


def getYear(dfAll):
    import numpy as np

    
    def custom1(y):
    
        years = np.linspace(1990,2020,31).astype(int).tolist()
        for i,year in enumerate(years):
            years[i] = str(year)
            if years[i] in y:
                #print(years[i])
                return years[i]
    
    
    dfAll['year'] = ''
    dfAll['year'] = dfAll['jref'].apply(lambda y: custom1(y))
    
    dfAll = dfAll[dfAll['year'] != None]
    dfAll = dfAll.dropna(axis = 0, subset =['year'])
    dfAll['year'] = dfAll['year'].astype(str)
    
    return dfAll

    
def getTok2(dfIn):
    
    tokens = []
    #start = time.time()
    #disable = ["tagger", "parser","ner","textcat"]
    for doc in nlp.pipe(df['title'].astype('unicode').values):#,disable = disable):
        #doc.text = doc.text.lower()
        #tokens.append([token.text.lower() for token in doc if not token.is_stop])
        
        tempList = ([token.lemma_ for token in doc if not token.is_stop])
    
    
        for i,string in enumerate(tempList):
            try:
                tempList[i] = string.lower()
            except:
                print(string)
    
        tokens.append(tempList)
    
    dfIn['tokens'] = tokens
    
    return dfIn


def updateBowDict(tokens,dicBow):
    ##### processing words
    ##### update dict with dic[word] = 0
    #tokens = processTxt(textIn)
    for tok in tokens:
        if tok not in dicBow:
            dicBow[tok] = 0
            
        elif tok in dicBow:
            dicBow[tok] = dicBow[tok] + 1
            #print(dicBow[tok])
    return dicBow
    
    
    
def setUpBow2(dfIn, extraWords):  #### adding the extra words to the list after the fact
    # transfering data in the df to dictionary formatt, getting tokens or lemma of the title
    # adding other info like year published and category to the tokens
    dicBow = {} ##### the dictionary storing the words for the BOW
    dicMain = {} #### dictionary with the row (or paper) id as the key, times cited, tokens, and number of tokens to be actually used in the model as values
    print('starting on getting toks')
    dfIn = getTok2(dfIn)  ##### getting tokens (or lemma) and adding the list as a new col in the df. # this is fairly slow
    print('got tokens')

    def custom2(x):
        #### adds to the list of toks in the tokens column
        ### extra info not in the title is added so it can be used in the bag of words
        tempList= []
        for col in extraWords:
            if col == 'cat':
                tempList = tempList + x[col].split(' ')

                
            elif col =='year':
                
                tempList = tempList +  [x[col]]
        
        return x['tokens'] +  tempList


    dfIn['tokens'] = dfIn.apply(custom2, axis=1)

    for i,row in dfIn.iterrows(): ###### looping through the dataframe (now with tokens/lemma) and building the dictionaries
        if i%2000 == 0:
            print(i)
        
        
        dicBow = updateBowDict(row['tokens'],dicBow)  #### update the bag of words dictionary 
        ##### add the extraWords in here
        
        ### updating the dict with row id, tokens, and cited by info
        dicMain[str(row.ids)] = [row.cited,row['tokens'],0]#### the zero will be updated with the word count for that ids later

    return dicBow, dicMain                                 


def removeFromDict(removeSet,dicIn): #### removes things add hoc from the dictionary
    #my_dict.pop('key', None) https://stackoverflow.com/questions/11277432/how-can-i-remove-a-key-from-a-python-dictionary
    for thing in removeSet:
        dicIn.pop(thing,None)

    return dicIn

def customProcTok(bowDic,symbol2Rem):
    #### custom text parsing to remove some of the trash that made it through spacy tokenization
    ### loop through the dicBow
    ### if key has len 1 remove it
    ### if it has any of the characters in symbol2rem
    ### if it can be converted into a float
    ### more than 2 characters of the string can be converted into a float
    okSet = set(['1d','2d','3d','4d'])
    tempDic = {}
    for key in bowDic:
        count = 0
        write = True  
        
        for n in key:
            try:
                float(n)
                count = count +1
            except:
                pass

            
            if n in symbol2Rem or count > 1:
                write = False
                break


        try:
            float(key)
            write = False
            continue
        except:
            pass

        if len(key) < 1:
            write = False
            continue    
        
        if len(key) < 3 and key not in okSet:
            write = False
            continue  
        


        if write == True:
            tempDic[key] = bowDic[key]

    return tempDic
    
def remTok(bowDic,set2Rem): ### i guess theres another ad hoc function to remove things from a dict
    
    for thing in set2Rem:
        try:
            bowDic.pop(thing,None)
        except:
            pass
    
    return bowDic
    

def popBow(dicBow,dicMain): ####### this is the main loop which contructs the one hot encoded matrix for input to
    # various models
    ## constructing a scipy sparse matrix


    label = [] #### label is the number of citations, what I will try to predict later

    
    bowVec = lil_matrix((len(dicMain), len(dicBow)), dtype=np.int8) ## this will hold the one hot encoded bow
    
    dicWord2Vec = {} ### keeps track of which word corresponds to which col in bowVec
    wordVec = np.zeros(len(dicBow))  ### is the same length as the num of words in the bow
    
    ########adding another dict which keeps track of which papers have which words
    dicWordPaper = dicBow.copy()  #### the keys are the words, values list of paper ids
    
    for i,key in enumerate(dicBow): 
        dicWord2Vec[key] = i #putting the location of each word into the dict
        dicWordPaper[key] = [] ### changing the value to emptylists
    
    indi = -1
    for key in dicMain: ################## loop used to populate bowVec, as well update dicMain with the number of factors for each row (paper) that are being used in the model
        indi = indi +1
        dicBowTemp = dicBow.copy() ### getting copy of the bow and setting all keys to zero, will use this to count words for the current row
        dicBowTemp = dict.fromkeys(dicBowTemp, 0) #setting all keys to 0 https://stackoverflow.com/questions/13712229/simultaneously-replacing-all-values-of-a-dictionary-to-zero-python
        
        wordVecTemp = wordVec.copy()
        label.append(dicMain[key][0]) ### label is the number of citations, what I will try to predict later
        tempSet = set()
        for tok in dicMain[key][1]: ### looping through the tokens stored in dicMain
            
            try: #### the token may have been removed as a model input, so I need to check if it's still in dicBow
                dicBowTemp[tok] = dicBowTemp[tok] + 1 #### #times the word is present
                
                dicWordPaper[tok].append(key)  ###### adding the id of the paper which has that word
                tempSet.add(tok) ###adding the token to a set which will be used to update the sparse array later
                dicMain[key][2] = dicMain[key][2] + 1 #### keeping track of how many words from this ids are still going to be used
            except:
                pass
        
        for item in tempSet:  ### looping through the tokens encountered for this row (paper) and updating the one hot encoding 

            indj = dicWord2Vec[item] ## getting the index (or column in the sparse matrix)
            try: #### at one point I was having trouble with indexing, so that's why its a try block
                bowVec[indi,indj] = dicBowTemp[item]
                
                if dicBowTemp[item] < 0:
                    print('warning! negative word count in dicBowTemp')
                    
                if bowVec[indi,indj] < 0:
                    print('waring! negative word count in bowVec')
                    
            except:
                print(str(indi) + ' ' + str(indj))
        
        # for j,key in enumerate(dicBowTemp): ############## this is the scaling problem.
        # # only need to non zero ones
        #     wordVecTemp[j] = dicBowTemp[key]
        
        #bowVec.append(wordVecTemp)

    return label,bowVec,dicWord2Vec,dicWordPaper  #### dicWordPaper is obsolete in this version


def analyzeFreq(dicIn):   
    ## this is also obsolete now
    
    tempDic = dicIn.copy()
    tempList= []
    #empty df with cols
    #https://www.kite.com/python/answers/how-to-create-an-empty-dataframe-with-column-names-in-python#:~:text=Use%20pandas.,an%20empty%20DataFrame%20with%20column_names%20.
    
    column_names = ["word", "count", "ids"]
    tempdf = pd.DataFrame(columns = column_names)
    
    
    for key in dicIn:
        tempDic[key] = len(dicIn[key])
        tempList.append(len(dicIn[key]))
        
        
        
    return tempDic,tempList


def histList(listIn): #makes histogram of list
    import numpy as np
    # import random
    from matplotlib import pyplot as plt
    
    # data = np.random.normal(0, 20, 1000) 
    
    # fixed bin size
    bins = np.arange(1, 100, 2) # fixed bin size
    
    plt.xlim([min(listIn)-5, max(listIn)+5])
    
    plt.hist(listIn, bins=bins, alpha=0.5)
    plt.title('Random Gaussian data (fixed bin size)')
    plt.xlabel('variable X (bin size = 5)')
    plt.ylabel('count')
    
    plt.show()

    
def trimBow2(bowDic,thresh):  ### removes words from bag of words if they present in less than thresh rows (papers)
    tempDic = {}
    for key in bowDic:
        if bowDic[key] > thresh:
            #bowDic.pop(key,None)
            tempDic[key] = bowDic[key]
            
    return tempDic


def wordCount(curBowDic,mainDic): #### i think this is also obsolete now
    #checks to see how many of the words for each paper are in the current bow dic.
    # need to know if have eliminated all or most words from anything
    
    tempDic = mainDic.copy()
    
    for key in mainDic:
        #print(key)
        count = 0
        for tok in mainDic[key][1]:
            if tok in curBowDic:
                count = count +1
        
        
        tempDic[key] = [mainDic[key][0],mainDic[key][1],count]
            
    return tempDic
        



def saveOutput2(fListIn,NListIn,outLoc):
        ###pickles the outfile and saves it to a dir
    import pickle
    import os
    
    #wrkDir = os.getcwd()
    #'relative/path/to/file/you/want'
    #os.chdir('')
    fileNames = NListIn
    vars2Save = fListIn
    for i,fileName in enumerate(fileNames):
        fileName = fileName + '.pckl'
        path = outLoc + '/' + fileName
        with open(path, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(vars2Save[i], f)
        f.close()



def prepToks(dfIn,extraWords):
    #adds the extra words to the title page
    #https://stackoverflow.com/questions/13331698/how-to-apply-a-function-to-two-columns-of-pandas-dataframe
    
    def custom1(x):
        tempStr = str()
        for col in extraWords:
            if col == 'cat':
                tempList = x[col].split(' ')
                for thing in tempList:
                    tempStr = tempStr + ' ' + str(thing)
                
            elif col =='year':
                
                tempStr = tempStr + ' ' +  x[col]
        
        return x['title'] + ' ' + tempStr
    
    df['title'] = df.apply(custom1, axis=1)
    
    return df

def checkIds(dfIn):  ##### ad-hoc function that was used to make sure there were now errors in converting ids to strings (ex 0704.237999999999999)
    maxLen = 0
    
    for i,row in dfIn.iterrows():
        maxLen = max(maxLen,len(row['ids']))
    
    return maxLen


def dropCust1(dfIn):
    ### removes the rows for which 'cited' cant be converted to an int
    def testIfInt(thing):
        
        try:
            int(thing)
            return True
        except:
            return False
    
    
    for index, row in dfIn.iterrows():
        if testIfInt(row['cited']) == False:
            dfIn.drop(index, inplace=True)

    return dfIn


if __name__ == "__main__":
    ############ load the data, keeping the ids col as a str
    df = pd.read_csv('processed_2-3.csv', dtype={'ids': object}) 
    df['ids'] = df['ids'].astype('str')
    df = dropCust1(df)
    print('loaded df')
    df = getYear(df) #### should already have the year
    print('got year')
    ##### want to go from jref to journal 
    #### is thr journal name on the gs result page somewhere?
    
    ### loading spacy library and stopwords
    nlp = spacy.load('en_core_web_sm')

    spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
    
    #toRemove = ['-']
    customize_stop_words = ['-','Single']  ##### adding custom stopwords
    for w in customize_stop_words:
        nlp.vocab[w].is_stop = True

    extraWords = ['year','cat','submit'] ####### cols from the df to be added to the bow
    
    toks2Rem = ['\n  ','--',"",',',"\to","..."]
    
    symbol2Rem = set(['%','$','{','}',"^","/", "\\",'#','*',"'",\
                  "''", '_','(',')', '..',"+",'-',']','['])
    
    start = time.time()
    dicBow,dicMain = setUpBow2(df,extraWords) ### get initial data
    end = time.time()### it doesn't look like I acutally use tok set
    
    setUpBow1 = end - start
    
    dicBow = customProcTok(dicBow,symbol2Rem) #### this isn't working right
    print('did customProcTok')
    ####remove words with low counts here
    thresh = 4
    dicBow = trimBow2(dicBow,thresh)
    print('trimmed bow')
    
    toks2Rem = ['\n  ','--',"",',',"\to","..."]
    dicBow = remTok(dicBow,toks2Rem)
    
    start = time.time()
    print('starting popBow')
    label,bowVec,dicWord2Vec,dicWordPaper = popBow(dicBow,dicMain)
    print('finished pop Bow')
    end = time.time()
    popBow1 = end - start
        
    outLoc = 'preProcessingOut'

    fList = [dicBow,dicMain,label]
    nList = ['dicBow','dicMain','label']

    saveOutput2(fList,nList,outLoc)  ###### saving variables in the list with the desired names
    
    bowVec = bowVec.tocsc()
    path = outLoc + '/' + 'bowVecSpares.npz'  
    scipy.sparse.save_npz(path, bowVec)  ##### saving the sparse matrix
    
    
    
    
        
        
