# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 13:02:39 2021





@author: bcyk5
"""

import sys

import pickle as pckl
import pandas as pd
import os
from scraper_api import ScraperAPIClient
#from bs4 import BeautifulSoup
from datetime import datetime
import time

    
    
def saveOutFile2(outLoc,var2Save,fileName):
    ###pickles the outfile and saves it to a dir
    import pickle
    import os
    
    #wrkDir = os.getcwd()
    #'relative/path/to/file/you/want'
    #os.chdir('')
    
    fileName = fileName + '.pckl'
    path = outLoc + '/' + fileName
    with open(path, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(var2Save, f)
    f.close()

    #os.chdir(wrkDir)


def loadPickled2(outName,path,fileName):
    #import pickle
    fileName = path + fileName + '.pckl'
    with open(fileName,'rb') as f:
        outName = pckl.load(f)
    
    return outName

def importInput(fileName):
    
    df = pd.read_csv(fileName,low_memory=False,dtype={'ids': object})
    df['ids'] = df['ids'].astype(str)
    try:
        df = df.drop(labels = ['Unnamed: 0'],axis = 1,inplace = False)
    except:
        pass 
    
    df = df.astype(str)
    
    return df
    

def cleanTitle(title):
    
    title = title.replace('\n','')
    title = title.replace('  ',' ')
    
    return title

    
def makeScrapInput2(dictIn,prevDone,num): ##### appending to df is too slow. 
### rewriting so all operations are with dictionaries

    wrkDir = os.getcwd()
    #os.chdir(prevDone)
    directory = wrkDir + '/' + prevDone + '/'
    outDic = {}
    returnDict = {}
    outList = []
    for filename in os.listdir(directory):
        if filename.endswith(".pckl"): 
            outName = filename.replace(".pckl","")
            
            outList.append(loadPickled2(outName,directory,outName))
            
    #merging dict
    #https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression-in-python-taking-union-o
    #https://www.tutorialspoint.com/python3/dictionary_update.htm
    
    for dic in outList:
        outDic.update(dic)
    
    #### loop through the df, if that row isn't allready in the dic
    #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.append.html
    # column_names = ["ids", "jref", "title"]
    # dfOut = pd.DataFrame(columns = column_names)
    count = 0
    
    for i,key in enumerate(dictIn):
        if count > num:   #### only loads num amount of data points for scrapping
            break
        
        if dictIn[key]['ids'] not in outDic:
            returnDict[count] = dictIn[key]
            count= count+1
            
    # for i, row in df.iterrows():
    #     if count>num:
    #         break
        
    #     if row['ids'] not in outDic:
    #         dfOut = dfOut.append(row)
    #         count = count +1
            
            
    return returnDict


    # look in the folder of stuff thats already been done
    # import the pickled dicts (data thats already scrapped needs to be saved as dict) 
    # cat dicts into one
    #loop through df, if id not in dict, add to new df or lists
    # when hit limit, stop
    
    #return df to be used as input
    
    


def getCitedBy(textIn):
    #need to grab the first occurance of cited by
    ind = textIn.index('Cited by')
    ind2 = ind + len('Cited by')
    
    ind3 = textIn.index('</a>',ind,-1)
    
    return textIn[ind:ind3]
    
    #return None


    
    
def checkTitle2(textIn):
    ## in the "oldRes"  Calculation of prompt diphoton production cross sections at Fermilab Tevatron and CERN LHC energies
    #title comes a bit after data-clk-atid
    
    #can use this to grab a small substring, then search for partial matches
    C1 = textIn.index('Cited by ')
    
    Tguess = textIn.rfind('data-clk-atid',0,C1)
    
    tempStr = textIn[Tguess:C1]
    
    indTag = tempStr.index('</a>')
    
    return tempStr[0:indTag]

def checkTitle3(textIn):
    ###V3 also returns some extra text which may or may not be useful later.
    ## in the "oldRes"  Calculation of prompt diphoton production cross sections at Fermilab Tevatron and CERN LHC energies
    #title comes a bit after data-clk-atid
    
    #can use this to grab a small substring, then search for partial matches
    C1 = textIn.index('Cited by ')
    
    Tguess = textIn.rfind('data-clk-atid',0,C1)
    
    tempStr = textIn[Tguess:C1]
    
    indTag = tempStr.index('</a>')
    
    return tempStr[0:indTag], tempStr
    
    
    
def makeUrl(strIn):
    ## just replace all the spaces with "+" and return the string
    #url template =https://scholar.google.com/scholar?q=Astrophys.J.679:1272-1287,2008&hl=en&as_sdt=0,48

    tempStr =  strIn.replace(' ','+')
    
    return 'https://scholar.google.com/scholar?q=' + tempStr + '&hl=en&as_sdt=0,48'
    
def popList(listIn,val):
    for i in range(len(listIn)):
        listIn[i] = val
        
    return listIn

def mainLoop(dfIn,saveDir,client,chckPntNum):
    
    def setupSave(saveDir,outDict):
        now = datetime.now()
        dt = now.strftime("%d/%m/%Y %H:%M:%S")
        dt = dt.replace('/','-')
        dt = dt.replace(':','_')
        saveOutFile2(saveDir,outDict,dt)
        
    outDict = {}
    sleep = False
    calls = 0
    for i, row in dfIn.iterrows():
        print(row['title'])
        if i%chckPntNum== 0:
            setupSave(saveDir,outDict) ### saving every 1000 times in case something crashes
            outDict = {}
            
        tempList = [[],[],[],[],[]]
        if sleep == True:
            time.sleep(0.3)
        try:
            
            url = makeUrl(row['jref'])
            
            result = client.get(url = url)
            calls = calls +1
            resStr = str(result)
            
            if resStr == '<Response [200]>':
                tempList[0] = row['title']
                tempList[1] = row['jref']
                tempList[2] = getCitedBy(result.text)
                #### modifying so that the extra text is also saved in the dict
                titleStr,extraTxt = checkTitle3(result.text)
                tempList[3] = titleStr
                tempList[4] = extraTxt
                #tempList[3] = checkTitle2(result.text)
                ###do the normal stuff
                #tempList = some functions
                outDict[row['ids']] = tempList 
                pass 
                
            elif resStr == '<Response [403]>':
                print('out of API requests!!!!!!!!!!!')
                break
            elif resStr == '<Response [429]>':
                print('requesting too fast')
                sleep = True
            
            else:
                tempList = popList(tempList,resStr) 
                outDict[row['id']] = tempList 
        
        except:

            print('exception')
            error = str(("Unexpected error: " +  str(sys.exc_info()[0])))
            print(error)
            
            tempList = popList(tempList,error) 
            outDict[row['ids']] = tempList 
            

        
        

    setupSave(saveDir,outDict)
    print('num API calls is ' + str(calls))
    return outDict,calls


#testing Phys.Lett.B653:434-438,2007
#url template =https://scholar.google.com/scholar?q=Astrophys.J.679:1272-1287,2008&hl=en&as_sdt=0,48

def findStrDF(df,col,strIn):
    return None

def findStrList(listIn,strIn):
    outList = []
    
    for thing in listIn:
        if strIn in thing:
            outList.append(thing)
            
    return outList

def dateHist(jrefs,dates):
    # outList = []
    # for i in range(len(dates)):
    #     outList.append([])

    outDict = {}
    for i,item in enumerate(dates):
        outDict[item] = 0


    for thing in jrefs:
        for t in dates:
            if str(t) in thing:
                outDict[t] = outDict[t] +1
                break 
    return outDict

def testErrorCode(url):
    

    client = ScraperAPIClient('XXXXXXXXXXXXXXXXXXXXXX')
    return client.get(url = url)
    

def writeBasicTxt(strIn,fileName):
    fileName = fileName + ".txt"
    strIn = str(strIn)
    
    f = open(fileName, "a")
    f.write(strIn)
    f.close()

if __name__ == "__main__":
    saveDir = 'prevDone'
    client = ScraperAPIClient('XXXXXXXXXXXXXXXXXXXXXXXXX')
    ####
    num = 10**6
    chckPntNum = 2000
    for file in os.listdir(os.getcwd()):
        if file.endswith('.csv') and 'dfSplitInput' in file:
            fileName = file
            
    df = importInput(fileName)
    df = df.astype(str)
    dictIn = df.to_dict('index')
    
    
    returnedDict = makeScrapInput2(dictIn,saveDir,num)
    dfIn = pd.DataFrame(returnedDict).transpose()
    del df
    start_time = time.time()
    
    #call setup funcs
    #make saveDir if not there
    
    outDict,calls = mainLoop(dfIn,saveDir,client,chckPntNum)
    
    exeTime = (time.time() - start_time)
    
    print('exe time was ' + str(exeTime))
    
    
    
    now = datetime.now()
    dt = now.strftime("%d/%m/%Y %H:%M:%S")
    dt = dt.replace('/','-')
    dt = dt.replace(':','_')
    
    fileName = 'runTime ' + dt
    
    writeBasicTxt(exeTime,fileName)
    

##################### functions written for ad-hoc checking of results and looking for errors
def adHocFindErr(dic):
    outList = []
    for key in dic:
        if "Unexpected error: <class 'ValueError'>" in dic[key][0]:
            outList.append(key)
            
    return outList
        

def findErIds(listIn,df):
    listOut = []
    for i, row in df.iterrows():
        for ID in listIn:
            if ID == row['ids']:
                listOut.append([row['ids'],row['jref'],row['title']])
                
                
    return listOut
############## end functions written for ad-hoc checking.........
            
            
            
            
            
