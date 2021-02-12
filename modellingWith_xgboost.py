# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 20:20:04 2021

first attempt at building a model to predict citations, using xgboost

predictions seem to mostly be the average of the target => means need to get more data, or more 
features which are useful in predicting citations

the current bag of words approach is very kludgy. it is also missing the info from:
    journal the paper was published in. This is likely to be very important
    authors other than sumbitter
    all institutions/universities involved. Some info on this is available in the raw output from the scapper, haven't gone though it in detail yet'
    info contained in abstract, will likely need more advanced nlp methods to deal with this

@author: bcyk5
"""

import pandas as pd


import xgboost as xgb

import numpy as np

import matplotlib.pyplot as plt
import numpy as np
from numpy import savez_compressed
from numpy import load
import scipy.sparse
from scipy.sparse import coo_matrix, lil_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


def loadPickled2(path):
    import pickle as pckl
    import os
    dummies = []
    path = os.getcwd() + '/' + path +'/'
    
    wrkDir = os.getcwd()
    os.chdir(path)
    
    for file in os.listdir(path):
        print(file)
        if file.endswith('.pckl'):

            with open(file,'rb') as f:
                dummies.append(pckl.load(f))
                f.close()
            
    os.chdir(wrkDir)
    return dummies ####### returning a list of all the variables
    
def loadFiles2(path):
    #gets the npzs, this probably should be combined with loadpickled by making them polymorphic. do if have time later
    import os

    dummies = []
    #fileName = path + fileName + '.pckl'
    path = os.getcwd() + '/' + path +'/'

    wrkDir = os.getcwd()
    os.chdir(path)
    
    for file in os.listdir(path):
        #print(file)
        if file.endswith('.npz'):
            with open(file,'rb') as f:
                dummies.append(scipy.sparse.load_npz(f))
                f.close()

    os.chdir(wrkDir)
            
    return dummies #### return list of variables loaded

    
def loadInputData(): ### ad hoc function to load data
    dirToRead = 'preProcessingOut 2-8'
    
    inputs = loadPickled2(dirToRead)
    X = loadFiles2(dirToRead)[0]
    
    target = inputs[2]
    dicBow = inputs[0]
    dicMain = inputs[1]
    
    target = np.asarray(target)
    target = target.astype(int)
    #target = np.log(target)
    print('loaded inputs')
    
    y = np.asarray(target)
    #X = np.asarray(bowVec)
    y = target
    print('changed to np arrays')
    
    return X,y, inputs



def tryCrossVal(X,y,model): #### not used in current run
    kfold = KFold(n_splits=3)
    results = cross_val_score(model, X, y, cv=kfold, scoring = 'neg_root_mean_squared_error')
    print("Accuracy: %.2f%% (%.2f%%)" % (results.mean(), results.std()))

    return results

def meanPcntEr(preds,y_test): ###### also not used in current run
    e = 2.71828
    outPut = np.zeros(len(preds))
    
    #change back to cites from logcites
    y_testTemp = e**y_test
    predsTemp = e**preds
    outPut = (predsTemp - y_testTemp)/y_test*100
    return outPut
    
    
def manTTS(dicMain,X,y,citedList): #### setting up the train test split
    #doing it this way so I can keep track of the ids for each paper (row)
    #split hardcoded to 80/20 rn

    ind = int(np.round(len(y)*.8))
    
    keys = list(dicMain.keys())
    
    Xm_train = X[0:ind,:]
    Xm_test = X[ind:-1,:]
    
    ym_train = y[0:ind]
    ym_test = y[ind:-1]
    
    keys_train = keys[0:ind]
    keys_test = keys[ind:-1]
    
    cited_train = citedList[0:ind]
    cited_test = citedList[ind:-1]
    
    return Xm_train,Xm_test,ym_train,ym_test,keys_train,keys_test,cited_train,cited_test
    


def getListCitedBy(dicMain):
    #### this gets a list of the citation numbers from the main dict. 
    tempList = []
    for thing in dicMain:
        tempList.append(int(dicMain[thing][0]))
    
    return tempList

def putResInDf(preds,cited_test,keys_test):

    #### puts the residuals back into the original df so that residuals can be analyzed
    
    from bagOfWords1stTry_2_4 import getYear

    def testIfInDic(thing):
        # if the ids for that row is the test_keys, return true, otherwise false
        try:
            keysDic[thing]
            return True
        except:
            return False
        
        
    def checkRow2(row):
        if int(keysDic[row['ids']][1]) == int(row['cited']): #### making sure that citations (target) from the data fed to the model is consistent with whats in the df. this is just a gravity check
            return int(row['cited']) - int(keysDic[row['ids']][0]) ### return the residual
        else:
            return np.nan #### if the data got mixed up somehow, return nan
            
        
            
    def ifInDic2(thing): #### checking to see if the ids for that row is in the dict
        try:
            keysDic[thing]
            return 'yes'
        except:
            return 'no'
            

    predActual = list(zip(preds, cited_test))  
    keysDic = dict(zip(keys_test, predActual))
    
    dfIn = pd.read_csv('processed_2-8.csv')  
    dfIn['ids'] = dfIn['ids'].astype('str')  
    #keysDic = dict(zip(keys_test, keys_test)) ####### making a dict out of the keys_test list
    # for faster lookup
    print('loaded df')
    dfIn = getYear(dfIn)
    
    
    # apply is way faster than iterrows    
    dfIn['temp'] = dfIn.apply(lambda x: ifInDic2(x.ids),axis = 1)
    
    dfIn = dfIn[dfIn['temp'] == 'yes'] ##### dropping all the rows that weren't in the test set

    print('dropped things not in keys_test')
    dfIn['res'] = 0
    
    ####### there are a few rows from the test set which don't match up with the data frame
    ####### this may be a formatting issue, or i loaded an older version of the df which is missing some data
    missingIds = []
    for i,row in dfIn.iterrows():
        try:
            keysDic[row['ids']]
        except:
            missingIds.append(row['ids'])
            
    ###### the line below calculates the residuals column, provided data is consistent
    dfIn['res'] = dfIn.apply(lambda x: checkRow2(x),axis = 1)
        
    print('added residuals')

    return dfIn, missingIds
    

if __name__ == '__main__':
    X,y,inputs = loadInputData()
    dicMain = inputs[1] ###### grabing the main dictionary from the list of loaded variables
    citedList = getListCitedBy(dicMain)
    
    ####### spliting up inputs
    Xm_train,Xm_test,ym_train,ym_test,keys_train,keys_test,cited_train,cited_test\
        = manTTS(dicMain,X,y,citedList) 
    
    data_dmatrix = xgb.DMatrix(data=Xm_train,label=ym_train) ### loading into xgboost format


    ########### running with k fold cross validation
    params = {"objective":"reg:squarederror",'colsample_bytree': 0.3,'learning_rate': 0.1,
                    'max_depth': 20, 'alpha': 10}
    
    cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                        num_boost_round=100,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
    
    
    print('did train test split, defined model')
    #cv_results.head()
    
    print(cv_results.tail(1))
    
    xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)
    print('trained')
    xgb.plot_importance(xg_reg,max_num_features=3)
    plt.rcParams['figure.figsize'] = [5, 5]
    plt.show()
    
    data_dmatrix_Xm_test = xgb.DMatrix(data=Xm_test)
    preds = xg_reg.predict(data_dmatrix_Xm_test)

    dfRes,missingIds = putResInDf(preds,cited_test,keys_test) #########putting the residuals back into the df for later analysis