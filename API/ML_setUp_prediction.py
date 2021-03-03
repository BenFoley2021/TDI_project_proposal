# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 14:43:47 2021

@author: bcyk5
"""
import keras
import os 
import spacy
import numpy as np
import pickle


class doML():
    """ an object which holds the model and methods for converting a texted input to one hot encoded input
    """
    def __init__(self):
        path = 'xxxxx'

        fileAndPath = path + 'zDic2WordVec.pckl'
        
        #need to load these on startup then pass them to functions as needed
        dictWord2Vec = pickle.load(open(fileAndPath, "rb" ))
        model = keras.models.load_model(path)
        
        self.model = model
        self.dictWord2Vec = dictWord2Vec
        
        print('called init')
        

    def setUpPredict(self, inputDict):
        """
        loads modules and files needed, calls words2ModelInput and predictOne
    
        """
    
        
        def predictOne(self, MLinput):
            """ return 
            """
            MLinput = np.reshape(MLinput, [1,16542]) ### need to reformat input. Ad-hoc s
        
            MLinput = MLinput.astype('int8')
            # dont want to load it everytime need to predict
            return self.model.predict(MLinput)
        
        def words2ModelInput(self,inputDict):
            """ converting a list of words to input for model
            
            """
            nlp = spacy.load('en_core_web_sm') #nlp called to get tokens
            X = np.zeros(len(self.dictWord2Vec))
            
            ### putting things together as one string
            allFact = inputDict['title'] + ' ' + inputDict['otherFactor']
            
            tokens = []
            factUsed = str()
            #
            tokens = ([token.lemma_ for token in nlp(allFact) if not token.is_stop])
            
            for word in tokens:
                word = word.strip().lower()
                print(word)
                #print(word)
                if word in self.dictWord2Vec:
                    X[self.dictWord2Vec[word]] += 1
                    factUsed += word + '_'
            return X, factUsed
        
        # inputDict = {'title' : 'energy blah blah de da nana'}
        # inputDict['otherFactor'] = ['2007', 'dumbass jounral', 'effect']
        
        MLinput, factUsed = words2ModelInput(self,inputDict)
        
        MLoutput = predictOne(self,MLinput)
        
        return MLoutput, factUsed
    



# inputDict = {'title' : 'energy blah blah de da nana'}
# inputDict['otherFactor'] = ['2007', 'dumbass jounral', 'effect']
# toML = setUpPredict(inputDict)