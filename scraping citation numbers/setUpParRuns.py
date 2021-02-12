# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 16:42:50 2021
##########

#spliting up inputs so that multiple instances of spyder can run each chunk of things to be scrapped
### the parameters for each individual run are controled by the script in each 


@author: bcyk5
"""

from possibleScrapAndVPN_3_2 import *
import time
import os

def splitInput(dictIn,numPar):
    import math
    import os
    cwd = os.getcwd()
    def makeDirs():
        dirs = []
        for i in range(numPar):
            dirs.append("block_" + str(i))
            path = cwd + '/' + "block_" + str(i) #### getting path for dir
            
            if os.path.isdir(path) == False: ### if the dir doesn't exist, make it
                os.mkdir(path)
            
            #os.mkdir(dirs[i])
        print(len(dirs))
        return dirs
        
        
    def saveSplitInput(tempDict,splitNum,dirs):
        dfTemp = pd.DataFrame(tempDict).transpose()
        
        path = dirs[splitNum]+ '/' + 'dfSplitInput' + str(splitNum) + '.csv'
        dfTemp.to_csv(path)
        
    def splitDict(dictIn,dirs):
        newL = math.ceil(len(dictIn)/numPar)
        print(newL)
        tempDict = {}

        splitNum = 0
        
        for i,key in enumerate(dictIn):
            if i < newL*(splitNum +1):
                tempDict[key] = dictIn[key]
                
            elif i>newL*(splitNum +1):
                
                tempDict[key] = dictIn[key]
                print(splitNum)
                print(i)
                saveSplitInput(tempDict,splitNum,dirs)
                splitNum = splitNum +1
                tempDict = {}
                
    dirs = makeDirs()
    splitDict(dictIn,dirs)

                

def setUpSubDir(script,dirName):
    from shutil import copyfile
    mainDir = os.getcwd()
    for dirc in os.listdir(mainDir):
        if 'block_' in dirc:
            print(dirc)
            copyfile(script, mainDir + '/' + dirc +'/'+ script)
        
            if dirName:
                os.mkdir(mainDir + '/' + dirc +'/'+ dirName)
                
                
def agResults(): 
    #### get all the current results into one dict, turn into df
    #### currently the results are stored as a dict with a list. 
    #### loop through folders, cat dicts
    #### list => dict => df
    from possibleScrapAndVPN_3_3 import loadPickled2
    mainDir = os.getcwd() 
    allDict = {} 
    for dirc in os.listdir(mainDir): 
        if 'block_' in dirc:
            path = dirc + '/' + 'prevDone' +'/'
            print(path) 
            
            for filename in os.listdir(path): 
                if filename.endswith(".pckl"):  
                    filename = filename.replace('.pckl','')
                    print(filename)
                    tempDict = loadPickled2('outDict',path,filename)
                    allDict.update(tempDict)
                    
    return allDict
                  
def outDict2df(outDict):
    ### change list to dict
    newDict = {} # this isn't space efficent by who cares
    ### keys: ids = ids, [0] = 'title', [1] = jref, [2] = cited,[3] = scrapTitle 
                    
    for i, key in enumerate(outDict):
        #print(str(i) + ' ' + key)
        tempDict = {'ids':key,'title':outDict[key][0],'jref':outDict[key][1],'cited':outDict[key][2],'scrapTitle':outDict[key][3]}
        newDict[i] = tempDict
        
        
    ####### converting to df
    dfOut = pd.DataFrame(newDict).transpose()
        
    return dfOut


def cleandfOut(dfOut):
    #### write some lambda functions to clean up the cited by col
    #### and drop the value errors
    #### should do a word analysis on the titles
    #### get the year column
    def getCited(x):
        
        out  =  x.replace('Cited by','')
        try:
            int(out)
        except:
            pass
        
        return out
    
    def getYear(y):
        import numpy as np
        years = np.linspace(1990,2020,31).astype(int).tolist()
        for i,year in enumerate(years):
            years[i] = str(year)
            if years[i] in y:
                #print(years[i])
                return years[i]

    
    
    dfOut['year'] = ''
    
    #### getting rid of all the data points where the scraper did work
    dfOut = dfOut[dfOut['title'] != "Unexpected error: <class 'ValueError'>"]
    
    dfOut['cited'] = dfOut['cited'].apply(lambda x: getCited(x))
    
    dfOut['year'] = dfOut['jref'].apply(lambda y: getYear(y))
    
    dfOut = dfOut[dfOut['year'] != None]
    
    return dfOut


def mergeOut(dfOut):
    ### need to merge the output df with the input so can have all the things
    #(year, submitter, cat) in the same df
    dfIn = pd.read_csv('df_02-2.csv',dtype={'ids': object})
    dfIn['ids'] = dfIn['ids'].astype('str')

    dfoutM = dfOut.merge(dfIn,left_on = 'ids',right_on = 'ids')
    
    #### remove extra cols
    toRem = ['Unnamed: 0','jref_y', 'title_y']
    
    dfoutM = dfoutM.drop(axis = 1, labels = toRem)
    dfoutM = dfoutM.rename(columns={"jref_x": "jref", "title_x": "title"})

    return dfoutM


if __name__ != "__main__":
    saveDir = 'prevDone'
    numPar = 25 #number of things to split up
    numDummy = 10**7
    #num = 50
    df = importInput('df_01-23.csv')
    df = df.astype(str)
    #dictIn = df.to_dict()
    dictIn = df.to_dict('index')
    
    
    ##### creating subDirs
    
    
    start_time = time.time()
    returnedDict = makeScrapInput2(dictIn,saveDir,numDummy)
    exeTime = (time.time() - start_time)
    
    print('exe time was ' + str(exeTime))
    
    splitInput(returnedDict,numPar)
    
    script = 'possibleScrapAndVPN_3_3.py'
    setUpSubDir(script,None)
    
    ### need to convert back to dict
    ###
    #dfIn = pd.DataFrame(returnedDict)    
    
    
############ this block of code is run to aggregate results from each sub-dir
# and combine them into an output which is passed to the analysis scripts
procRes = True

if procRes == True: ### setting up script to process output data
    
    allDic = agResults()
    dfOut = outDict2df(allDic)
    
    dfOut= cleandfOut(dfOut)
    
    dfOut = mergeOut(dfOut)
    
    print(len(dfOut['ids'].unique()))
    
    dfOut.to_csv('processed_2-8.csv')
    
    ### 2-4 11am, len dfOut = 135027, checked uniqueness of ids
########end this block of code is run to aggregate results..........
    