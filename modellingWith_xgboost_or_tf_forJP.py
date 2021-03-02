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


as of 2/23
keras runs, loss function decreases
to do next: how to I include validation data. I added dropouts to the layers (runs) but
dont know how to put the validation data in. from a quick search this may be non trivial
https://github.com/keras-team/keras/issues/2702
https://github.com/keras-team/keras/issues/10472

@author: bcyk5
"""

import pandas as pd
import math



import numpy as np

import matplotlib.pyplot as plt
import numpy as np
from numpy import savez_compressed
from numpy import load
import scipy.sparse
from scipy.sparse import coo_matrix, lil_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
#import tensorflow as tf

def loadPickled2(path):
    """ loads the pickled files in a specfied directory

    ----------
    path : string
        the subfolder to load files from

    Returns
    -------
    varList : list
        DESCRIPTION: list storing each variable that was found as a .pckl file
    nameList : list
        list of strings, names of files corresponding to the vars in varlist

    """
    import pickle as pckl
    import os
    varList = []
    nameList = []
    path = os.getcwd() + '/' + path +'/'
    
    wrkDir = os.getcwd()
    os.chdir(path)
    
    for file in os.listdir(path):
        print(file)
        if file.endswith('.pckl'):
            nameList.append(file)
            with open(file,'rb') as f:
                varList.append(pckl.load(f))
                f.close()
            
    os.chdir(wrkDir)
    return varList, nameList ####### returning a list of all the variables
    
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

    

def loadInputData2(): ### another ad hoc function to load the data for when the files needed to be split up becuase i can't upload things larger than 25 MB to github
    """ loads data for ML
    X = sparse matrix
    y = array of float, the citations per year
    inputs = list of y and dictionaries which have all the data used to 
    construct X
    """    

    dirToRead = 'preProcOut_2-27'
    
    inputs,names = loadPickled2(dirToRead)
    X = loadFiles2(dirToRead)[0]
    
    target = inputs[2]
    dicBow = inputs[0]
    dicMain = inputs[1]
    
    target = np.asarray(target)
    #target = np.log(target)
    print('loaded inputs')
    
    y = np.asarray(target)
    #X = np.asarray(bowVec)
    y = target
    print('changed to np arrays')
    
    return X,y, inputs,names

    
def manTTS(keys,X,y,citedList): #### setting up the train test split
    #doing it this way so I can keep track of the ids for each paper (row)
    #split hardcoded to 80/20 rn

    ind = int(np.round(len(y)*.8))
    
    #keys = list(dicMain.keys())
    
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


def setUp4Keras(Xm_train, Xm_test, ym_train, ym_test):
    """ setting up inputs to used with keras
    scaling and experimenting with ways to put the sparse matrix in
    
    """
    #### min max scaler does not accept sparse input
    # from sklearn.preprocessing import MinMaxScaler
    # scaler = MinMaxScaler()
    # Xm_train = scaler.fit_transform(Xm_train)
    # Xm_test = scaler.fit_transform(Xm_test)

    ##### creating td.sparse object

    #https://stackoverflow.com/questions/40896157/scipy-sparse-csr-matrix-to-tensorflow-sparsetensor-mini-batch-gradient-descent

    coo = Xm_train.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, ym_train, coo.shape)

def trainKerasVal(Xm_train, Xm_test, ym_train, ym_test):
    #redoing trainKeras but with validation generator
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    import tensorflow as tf
    from math import ceil
    
    def batch_generator(Xl, yl, batch_size): ####### should prob rewrite this
        number_of_batches = samplesPerEpoch/batchSize
        counter=0
        shuffle_index = np.arange(np.shape(yl)[0])
        np.random.shuffle(shuffle_index)
        Xl =  Xl[shuffle_index, :]
        yl =  yl[shuffle_index]
        while 1:
            index_batch = shuffle_index[batch_size*counter:batch_size*(counter+1)]
            X_batch = Xl[index_batch,:].todense()
            y_batch = yl[index_batch]
            counter += 1
            yield(np.array(X_batch),y_batch)
            if (counter < number_of_batches):
                np.random.shuffle(shuffle_index)
                counter=0
    
    
    model = Sequential()
    model.add(Dense(Xm_test.shape[1],activation = 'relu'))
    model.add(Dropout(.5))
    model.add(Dense(25,activation = 'relu'))
    model.add(Dropout(.3))
    model.add(Dense(10,activation = 'relu'))
    model.add(Dropout(.5))
    model.add(Dense(1))
    
    samplesPerEpoch=Xm_train.shape[0]
    batchSize = ceil(Xm_train.shape[0]/300)
    batchSizeV = ceil(Xm_test.shape[0]/300)
    
    # for log loss https://keras.io/api/losses/regression_losses/#mean_squared_logarithmic_error-function
    model.compile(optimizer = 'adam', loss = 'mean_squared_logarithmic_error')
    # model.fit_generator(generator = batch_generator, (Xm_train, ym_train, batch_size = batchSize),\
    #                     validation_data = batch_generator, (Xm_tets, ym_test, batch_size = batchSizeV), \
    #                     steps_per_epoch=1, epochs = 1)
    
    train_generator = batch_generator(Xm_train, ym_train, batch_size = batchSize)
    val_generator = batch_generator(Xm_test, ym_test, batch_size = batchSizeV)
    
    history = model.fit_generator(generator = train_generator, \
                        validation_data = val_generator, \
                        steps_per_epoch = 300, epochs = 5)
        
        
    return model, history

def trainKeras(Xm_train, Xm_test, ym_train, ym_test):
    """ training a keras model
    currently experimenting to get it to work with sparse inputs

    Parameters
    ----------
    Xm_train : TYPE
        DESCRIPTION.
    Xm_test : TYPE
        DESCRIPTION.
    ym_train : TYPE
        DESCRIPTION.
    ym_test : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    
    current problem:is out of order. Many sparse ops require sorted indices.
    Use `tf.sparse.reorder` to create a correctly ordered copy.
    https://stackoverflow.com/questions/61961042/indices201-0-8-is-out-of-order-many-sparse-ops-require-sorted-indices-use
    
    https://www.tensorflow.org/api_docs/python/tf/sparse/SparseTensor
    """    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    import tensorflow as tf
    from math import ceil
    
    def batch_generator(Xl, yl, batch_size): ####### should prob rewrite this
        number_of_batches = samplesPerEpoch/batchSize
        counter=0
        shuffle_index = np.arange(np.shape(yl)[0])
        np.random.shuffle(shuffle_index)
        Xl =  Xl[shuffle_index, :]
        yl =  yl[shuffle_index]
        while 1:
            index_batch = shuffle_index[batch_size*counter:batch_size*(counter+1)]
            X_batch = Xl[index_batch,:].todense()
            y_batch = yl[index_batch]
            counter += 1
            yield(np.array(X_batch),y_batch)
            if (counter < number_of_batches):
                np.random.shuffle(shuffle_index)
                counter=0
    
    
    model = Sequential()
    model.add(Dense(Xm_test.shape[1],activation = 'relu'))
    model.add(Dropout(.5))
    model.add(Dense(25,activation = 'relu'))
    model.add(Dropout(.3))
    model.add(Dense(10,activation = 'relu'))
    model.add(Dropout(.5))
    model.add(Dense(1))
    
    samplesPerEpoch=Xm_train.shape[0]
    batchSize = ceil(Xm_train.shape[0]/300)
    
    # for log loss https://keras.io/api/losses/regression_losses/#mean_squared_logarithmic_error-function
    model.compile(optimizer = 'adam', loss = 'mean_absolute_error')
    model.fit_generator(generator=batch_generator(Xm_train, ym_train, \
                    batch_size = batchSize), steps_per_epoch=300, epochs = 4)
    
    return model
        
def predictKeras(model,Xm):
    
    preds = []
    incSize = 5000
    numBatch = math.ceil(Xm_test.shape[0]/incSize)
    for ind in range(numBatch):
        if (ind + 1) * incSize > Xm_test.shape[0]:
            indEnd = Xm_test.shape[0]
        else:
            indEnd = (ind + 1) * incSize
        
        X_batch = Xm_test[(ind * incSize): indEnd, :].todense()
            
        preds.append(model.predict(X_batch))
    
    return preds

def saveResults(keys_test,preds,ym_test):
    #### basic file to save things, all it does is help me remeber
    #how to set up inputs for saveOutput2
    outLoc = 'fitting_2-25'
    from rawDataToBagOfWords_oneHotEncode_2 import saveOutput2


    fList = [keys_test,preds,ym_test]
    nList = ['keys_test','preds','ym_test']

    saveOutput2(fList,nList,outLoc)  ###### saving variables in the list with the desired names

def tempFlattenPreds(preds):
    # when prediction done in batches (memory issuses with large sparse)
    # need to flatten
    
    predsOut = []
    
    for ar in preds:
        for num in ar:
            predsOut.append(num[0])
            
    return predsOut

def packagePreds(preds,keys_test,ym_test,dicMain):
    #puts the preds and actual into a dict with the id as the key
    
    def verifyOrder(resDict,dicMain):
        misMatchList = []
        for key in resDict:
            if dicMain[key][0] != resDict[key][1]:
                misMatchList.append(key)
        return misMatchList
            
    resDict = {} # residuals dictionary
    
    for i, key in enumerate(keys_test):
        try:
            resDict[key] = [preds[i],ym_test[i]]
        except:
            print(str(i) + ' ' + key)
        
    misMatchList = verifyOrder(resDict,dicMain)
        
    return resDict, misMatchList
    
def shuffle(dicMain, X, y):
    from sklearn.utils import shuffle
    
    dicMainKeys = list(dicMain.keys())
    dicMainKeys = np.array(dicMainKeys)
    dicMainKeys, X, y = shuffle(dicMainKeys, X, y, random_state=0)
    
    return dicMainKeys, X, y
    
if __name__ == '__main__':
    from sklearn.metrics import mean_squared_log_error
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error

    X, y, inputs, names = loadInputData2()
    dicMain = inputs[1] ###### grabing the main dictionary from the list of loaded variables
    citedList = getListCitedBy(dicMain)
    
    dicMainKeys, X, y = shuffle(dicMain, X, y)
    
    #X = X.tocsr(copy=True)
    
    ####### spliting up inputs
    Xm_train, Xm_test, ym_train, ym_test, keys_train, keys_test, cited_train, \
        cited_test = manTTS(dicMainKeys, X, y, citedList) 
    
    ## min max scaler doesn't work
    #sparseInput = setUp4Keras(Xm_train, Xm_test, ym_train, ym_test)
    
    model = trainKeras(Xm_train, Xm_test, ym_train, ym_test)
    print(model.count_params())
    
    import os
    cwd = os.getcwd()
    model.save(cwd)
    
    preds = predictKeras(model,Xm_test)
    
    predsOut = tempFlattenPreds(preds)
    
    resDict, misplaced = packagePreds(predsOut,keys_test,ym_test,dicMain)
    
    mean_squared_log_error(ym_test, predsOut)
    
    from sklearn.utils import shuffle
    ym_test_shuffle = shuffle(ym_test, random_state=1)
    
    mean_squared_log_error(ym_test, ym_test_shuffle)
        
    mean_squared_error(ym_test, predsOut)
    
    mean_absolute_error(ym_test, predsOut)
    
    np.median(ym_test)
    
    import pickle
    pickle.dump(resDict, open( "resDict_2-26_MAE.p", "wb" ) )
    
    #putResInDf(preds,cited_test,keys_test,dicMain)
    
if __name__ != '__main__':
    ##### temp stuff, trying to save and relaod the model
    import keras
    
    cwd = os.getcwd()

    toLoad = cwd + "\\" + 'saved_model.pb'
    
    model = keras.models.load_model(cwd)
    
    preds = predictKeras(model,Xm_test)

    MLinput = Xm_test[0,:].todense()

    model.predict(MLinput)



    
