# -*- coding: utf-8 -*-
"""
Created on Sat Jul 08 10:30:42 2017

@author: Administrator

"""

import re
import math
import operator
import feedparser
from numpy import *

#==============================================================================
#词表到向量的转换函数 

def loadDataSet():
    postingList = [['my','dog','has','flea','problems','help','please'],
                   ['maybe','not','take','him','to','dog','park','stupid'],
                   ['my','dalmation','is','so','cute','I','love','him'],
                   ['stop','posting','stupid','worthless','garbage'],
                   ['mr','licks','ate','my','steak','how','to','stop','him'],
                   ['quit','buying','worthless','dog','food','stupid']]
    classVec = [0,1,0,1,0,1]  #1代表侮辱性文字，0代表正常言论
    return postingList,classVec

def createVocabList(dataSet):
    #创建一个空集
    vocabSet = set ([])
    for document in dataSet:
        #操作符|用于创建两个集合的并集
        vocabSet = vocabSet | set (document)
    return list(vocabSet)

def setOfWords2Vec(VocabList,inputSet):
    #创建一个其中所含元素都为0的向量
    returnVec = [0]*len(VocabList)
    for word in inputSet:
        if word in VocabList:
            returnVec[VocabList.index(word)] = 1
        else:
            print "the word:%s is not in my Vocabulary!" % word
    return returnVec

#==============================================================================
# 朴素贝叶斯的=分类器训练函数

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    #初始化概率
#==============================================================================
#     p0Num = zeros(numWords)
#     p1Num = zeros(numWords)
#     p0Denom = 0.0 ; p1Denom = 0.0
#==============================================================================
    p0Num = ones(numWords); p1Num = ones(numWords)
    p0Denom = 2.0; p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] ==1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive


#==============================================================================
#朴素贝叶斯分类函数            
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0
    
def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0v,p1v,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love','my','dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0v,p1v,pAb)
    testEntry = ['stupid','garbage']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0v,p1v,pAb)    

#==============================================================================
# 朴素贝叶斯词袋模型
def bagOfwords2VecMN(vocabList,inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec
   
#==============================================================================
# 文本解析及完整的垃圾邮件的测试函数
def textParse(bigString):
    listOfTokens = re.split(r'\W*',bigString)
    return [x.lower() for x in listOfTokens if len(x) > 2]

def spamTest():
    docList = []; classList = []; fullText = []
    #导入文件并解析文本
    for i in range(1,26):
        wordList = textParse(open(r'E:\ML\ML_source_code\mlia\Ch04\email\spam\%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open(r'E:\ML\ML_source_code\mlia\Ch04\email\ham\%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    traningSet = range(50); testSet = []
    #随机的构建训练集
    for i in range(10):
        randIndex = int(random.uniform(0,len(traningSet)))
        testSet.append(traningSet[randIndex])
        del(traningSet[randIndex])
        
    trainMat = []; trainClasses = []
    for docIndex in traningSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0v,p1v,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0v,p1v,pSpam) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is:',float(errorCount)/len(testSet)

    
#==============================================================================
# 使用RSS源分类器及高频词去除函数
ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')

def calcMostFre(vocabList,fullText):
    """
    函数遍历词汇表中出现的每个词语的次数，
    并排序，返回次数最高的30个词
    """
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(),key = operator.itemgetter(1),reverse = True)
    return sortedFreq[:30]

def localWords(feed1,feed0):
    docList = [];classList = [];fullText = []
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        #每次只去访问一条RSS源
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)      
    vocabList = createVocabList(docList)
    #去掉那些出现次数最高的那些词
    top30Words = calcMostFre(vocabList,fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = range(2*minLen); testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfwords2VecMN(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0v,p1v,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfwords2VecMN(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0v,p1v,pSpam) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is: ',float(errorCount)/len(testSet)
    return vocabList,p0v,p1v
        
#==============================================================================
# 最具表征性的词汇显示函数

def getTopWords(ny,sf):
    vocabList,p0v,p1v = localWords(ny,sf)
    topNY = []; topSF = []
    for i in range(len(p0v)):
        if p0v[i] > -6.0:
            topSF.append((vocabList[i],p0v[i]))
        if p1v[i] > -6.0:
            topNY.append((vocabList[i],p1v[i]))
    sortedSF = sorted(topSF,key = lambda pair: pair[1], reverse = True)
    print "SF**SF****SF**SF**SF****SF**SF**SF****SF**SF**SF****SF**SF**SF****SF**SF"
    
    for item in sortedSF:
        print item[0]
    sortedNY = sorted(topNY,key = lambda pair: pair[1], reverse = True)
    print "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY"
    
    for item in sortedNY:
        print item[0]    
        