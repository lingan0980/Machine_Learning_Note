# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 23:33:23 2017

@author: Administrator
"""

import numpy as np
from numpy import *
import matplotlib.pyplot as plt


#==============================================================================
# 标准回归函数和数据导入函数 

def loadDataSet(fileName):
    """
    函数能够自检出特征的数目
    """
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

#求最佳拟合直线
def standRegres(xArr,yArr):
    xMat = np.mat(xArr); yMat = np.mat(yArr)
    xTx = xMat.T * xMat 
    #判断矩阵是否可逆,np.linalg.det()矩阵求行列式（标量）
    if np.linalg.det(xTx) == 0.0:   
        print "This matrix is singular, cannot do inverse"
        return
    #ws = xTx.I * (xMat.T * yMat)        #矩阵的逆矩阵：A.I
    ws = np.linalg.solve(xTx, xMat.T * yMat.T)  
    return ws 


##################
#==============================================================================
# import matplotlib.pyplot as plt
# 
# xMat = np.mat(xArr)
# yMat = np.mat(yArr)
# yHat = xMat * ws
# 
# fig = plt.figure()
# ax = fig.add_subplot(111)  #111表示将画布划分为1行2列选择使用从上到下第一块  
# ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
# 
# 
# xCopy = xMat.copy()
# xCopy.sort(0)
# yHat = xCopy * ws
# ax.plot(xCopy[:, 1], yHat)
# plt.show()
# 
#np.corrcoef(yHat.T, yMat)  #相关系数：np.corrcoef(x,y)
#==============================================================================


#==============================================================================
# 局部加权线性回归函数
def lwlr(testPoint, xArr, yArr, k = 1.0):
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye((m)))  #创建对角权重矩阵
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = np.exp(diffMat * diffMat.T/(-2.0*k**2))  #权重大小以指数级衰减
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:   #矩阵行列式 
        print "This matrix is singular, cannot do inverse"
        return
    #ws = xTx.I * (xMat.T * (weights * yMat))
    ws = np.linalg.solve(xTx, xMat.T * (weights * yMat))
    return testPoint * ws 


def lwlrTest(testArr, xArr, yArr, k = 1.0):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

#==============================================================================
# xArr,yArr = regression.loadDataSet(r"E:\ML\ML_source_code\mlia\Ch08\ex0.txt")
# print "k=1.0 :"  ,lwlr(xArr[0], xArr, yArr, 1.0)
# print "k=0.001 :",lwlr(xArr[0], xArr, yArr, 1.0)
# print "k=0.003:" ,lwlr(xArr[0], xArr, yArr, 1.0)
# 
#==============================================================================

#==============================================================================
# yHat = regression.lwlrTest(xArr, xArr, yArr, 0.003)
# yHat = regression.lwlrTest(xArr, xArr, yArr, 0.001)
# yHat = regression.lwlrTest(xArr, xArr, yArr, 1.0)
# 
# xMat = np.mat(xArr)
# srtInd = xMat[:,1].argsort(0)
# xSort = xMat[srtInd][:, 0 ,:]
# 
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(xSort[:,1],yHat[srtInd])
# ax.scatter(xMat[:,1].flatten().A[0], np.mat(yArr).T.flatten().A[0],s=2,c='red')
# plt.show()
# 
#==============================================================================
#==============================================================================
# 预测鲍鱼的年龄
def rssError(yArr, yHatArr):
    return ((yArr - yHatArr)**2).sum()


#==============================================================================
# abX,abY = regression.loadDataSet(r'E:\ML\ML_source_code\mlia\Ch08\abalone.txt')
# yHat01 = regression.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
# yHat1 = regression.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1.0)
# yHat10 = regression.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
# 
# #分析预测误差大小
# regression.rssError(abY[0:99], yHat01.T)
# regression.rssError(abY[0:99], yHat1.T)
# regression.rssError(abY[0:99], yHat10.T)
# 
# yHat01 = regression.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
# regression.rssError(abY[100:199], yHat01.T)
# 
# yHat1 = regression.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1.0)
# regression.rssError(abY[100:199], yHat1.T)
# 
# yHat10 = regression.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
# regression.rssError(abY[100:199], yHat10.T)
# 
# ws = regression.standRegres(abX[0:99], abY[0:99])
# yHat = np.mat(abX[100:199]) * ws
# regression.rssError(abY[100:199], yHat.T.A)
# 
#==============================================================================

#岭回归
def ridgeRegres(xMat, yMat, lam = 0.2):
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1])*lam
    if np.linalg.det(xTx) == 0.0:   
        print "This matrix is singular, cannot do inverse"
        return
    ws = denom.I * (xMat.T * yMat)                           
    return ws

def ridgeTest(xArr, yArr):
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)                 
    yMat = yMat - yMean
    xMeans = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    xMat = (xMat - xMeans)/xVar
    numTestPts = 30
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, np.exp(i-10))
        wMat[i, :] = ws.T
    return wMat

#==============================================================================
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(ridgeWeights)
# plt.show()
#==============================================================================

def regularize(xMat):
    inMat = xMat.copy()
    inMeans = np.mean(inMat, 0)                 
    inVar = np.var(inMat, 0)
    xMat = (inMat - inMeans)/inVar
    return xMat


#向前逐步回归
def stageWise(xArr, yArr, eps = 0.1, numIt = 100):
    """
    xArr 输入数据; yArr 预测变量
    eps 每次迭代需要调整的步长; numIt 迭代次数
    """
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)                 
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m, n = np.shape(xMat)
    returnMat = np.zeros((numIt,n))
    ws = np.zeros((n, 1)); wsTest = ws.copy(); wsMax = ws.copy()
    for i in range(numIt):
        print ws.T
        lowestError = inf
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat





from time import sleep
import json
import urllib2

import requests
import urllib.request
resp = requests.get(searchURL)

def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    sleep(10)
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    #searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum)

    searchURL = r'E:\ML\ML_source_code\mlia\Ch08\setHtml\lego8288.html' % (myAPIstr, setNum)

    pg = urllib2.urlopen(searchURL).read()
    retDict = json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else: newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if  sellingPrice > origPrc * 0.5:
                    print "%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc, sellingPrice)
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except: print 'problem with item %d' % i
    
def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
#==============================================================================
#     searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
#     searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
#     searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
#     searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
#     searchForSet(retX, retY, 10196, 2009, 3263, 249.99)
#==============================================================================
    
def crossValidation(xArr,yArr,numVal=10):
    m = len(yArr)                           
    indexList = range(m)
    errorMat = zeros((numVal,30))#create error mat 30columns numVal rows
    for i in range(numVal):
        trainX=[]; trainY=[]
        testX = []; testY = []
        random.shuffle(indexList)
        for j in range(m):#create training set based on first 90% of values in indexList
            if j < m*0.9: 
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX,trainY)    #get 30 weight vectors from ridge
        for k in range(30):#loop over all of the ridge estimates
            matTestX = mat(testX); matTrainX=mat(trainX)
            meanTrain = mean(matTrainX,0)
            varTrain = var(matTrainX,0)
            matTestX = (matTestX-meanTrain)/varTrain #regularize test with training params
            yEst = matTestX * mat(wMat[k,:]).T + mean(trainY)#test ridge results and store
            errorMat[i,k]=rssError(yEst.T.A,array(testY))
            #print errorMat[i,k]
    meanErrors = mean(errorMat,0)#calc avg performance of the different ridge weight vectors
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors==minMean)]
    #can unregularize to get model
    #when we regularized we wrote Xreg = (x-meanX)/var(x)
    #we can now write in terms of x not Xreg:  x*w/var(x) - meanX/var(x) +meanY
    xMat = mat(xArr); yMat=mat(yArr).T
    meanX = mean(xMat,0); varX = var(xMat,0)
    unReg = bestWeights/varX
    print "the best model from Ridge Regression is:\n",unReg
    print "with constant term: ",-1*sum(multiply(meanX,unReg)) + mean(yMat)












