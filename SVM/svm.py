# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 11:57:31 2017

@author: Administrator
"""

from numpy import *
import numpy as np
import operator
from os import listdir


#==============================================================================
# SMO算法中的辅助函数
def loadDataSet(fileName):
    fileName = r"E:\ML\ML_source_code\mlia\Ch06\testSet.txt"
    dataMat = [];labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def selectJrand(i,m):
    """
    i 是第一个alpha下标，m是所有alpha的数目
    """
    j = i
    while (j == i):
        j = int(np.random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

"""
#SMO函数的伪代码
创建一个alpha向量并将其初始化为0向量
当迭代次数小于最大迭代次数时(外循环):
    对数据集中的每个数据向量(内循环):
        如果该数据向量可以被优化:
            随机选择另外一个数据向量
            同时优化这两个向量
            如果两个向量都不能被优化，退出内循环
    如果所有的向量都没有被优化，增加迭代项目，继续下一次循环
"""

#==============================================================================
# 简化版SMO算法
def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    """
    函数(数据集,类别标签,常数C,容错率,退出前最大的循环次数)
    """
    dataMatrix = np.mat(dataMatIn); labelMat = np.mat(classLabels).transpose()
    b = 0; m,n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m,1)))
    iter = 0
    while (iter < maxIter):
        aphaPairsChanged = 0
        for i in range(m):
            fXi = float(np.multiply(alphas,labelMat).T * (dataMatrix*dataMatrix[i,:].T)) + b
            Ei = fXi - float(labelMat[i])
            #如果alpha可以更改进入优化过程
            if ((labelMat[i]*Ei<-toler) and (alphas[i]<C)) or ((labelMat[i]*Ei>toler) and (alphas[i]>0)):
                #随机选择第二个alpha
                j = selectJrand(i,m)
                fXj = float(np.multiply(alphas,labelMat).T * (dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy();
                alphaJold = alphas[i].copy();
                #保证alpha在0与C之间
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C +  alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C ,alphas[j] + alphas[i])
                if L==H:
                    print "L==H";continue
                eta = 2.0*dataMatrix[i,:]*dataMatrix[j,:].T-dataMatrix[i,:]*dataMatrix[i,:].T-dataMatrix[j,:]*dataMatrix[j,:].T
                if eta > 0:
                    print "eta>0";continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print "j not moving enough";continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
                #对i进行修改，修改量与j相同，但方向相反
                b1 = b - Ei - labelMat[i]*(alphas[i] - alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j] - alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej - labelMat[i]*(alphas[i] - alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j] - alphaIold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0<alphas[i]) and (C>alphas[i]):
                    b = b1
                elif (0<alphas[j]) and (C>alphas[j]):
                    b = b2
                else:
                    b = (b1+b2)/2.0
                aphaPairsChanged += 1
                print "iter: %d i: %d,pairs changed %d" %(iter,i,aphaPairsChanged)
        if (aphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print "iteration number: %d" % iter
    return b,alphas

#==============================================================================
# 完整版的Platt SMO算法
#==============================================================================
# class optStruct:
#     def __init__(self, dataMatIn, classLabels, C, toler):
#         self.X = dataMatIn
#         self.labelMat = classLabels
#         self.C = C
#         self.tol = toler
#         self.m = np.shape(dataMatIn)[0]                    
#         self.alphas = np.mat(np.zeros((self.m, 1)))
#         self.b = 0
#         #误差缓存
#         self.eCache = np.mat(np.zeros((self.m, 2)))
#==============================================================================
  
class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        #记住是双括号
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[: , i] = kernelTrans(self.X, self.X[i, :], kTup)
      
def calcEk(oS,k):
    #fXk = float(np.multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k,:].T)) + oS.b
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

    #内循环中的启发式方法
def selectJ(i,oS,Ei):
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]
    #构建了一个非零表
    validEcacheList = np.nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:continue
            Ek = calcEk(oS,k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                #选择具有最大步长的j
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK,Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS,j)
    return j,Ej

def updateEk(oS,k):
    Ek = calcEk(oS,k)
    oS.eCache[k] = [1,Ek]
        
#==============================================================================
# 完整的Platt SMO算法中的优化例程
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        #第二个alpha选择中的启发式方法
        j,Ej = selectJ(i,oS,Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C,oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: 
            print "L==H"; return 0
        #eta = 2.0 * oS.X[i,:] * oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - oS.X[j,:]*oS.X[j,:].T
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]                        
        if eta >= 0:
            print "eta>=0"; return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        #更新误差缓存
        updateEk(oS,j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print "j not moving enough"; return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
        #更新误差缓存
        updateEk(oS,i)
        b1 = oS.b-Ei-oS.labelMat[i]*(oS.alphas[i] - alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b-Ej-oS.labelMat[i]*(oS.alphas[i] - alphaIold)*oS.K[i,j] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
#==============================================================================
#         b1 = oS.b-Ei-oS.labelMat[i]*(oS.alphas[i] - alphaIold)*oS.X[i,:]*oS.X[i,:].T-oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
#         b2 = oS.b-Ej-oS.labelMat[i]*(oS.alphas[i] - alphaIold)*oS.X[i,:]*oS.X[j,:].T-oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
#==============================================================================
        if (0< oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0< oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        return 1
    else:
        return 0
    """
    使用了自己的数据结构，该结构的在参数oS中传递
    """
    
#==============================================================================
# 完整版platt SMO的外循环代码

def smoP(dataMatIn,classLabels,C,toler,maxIter,kTup=('lin',0)):
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(),C,toler,kTup)
    iter = 0
    entireSet = True; aphaPairsChanged = 0
    while (iter < maxIter) and ((aphaPairsChanged > 0) or (entireSet)):
        aphaPairsChanged = 0
        #遍历数据集中所有的值
        if entireSet:
            for i in range(oS.m):
                aphaPairsChanged += innerL(i,oS)
                print "fullSet, iter:  %d i,%d, pairs changed %d" %(iter,i,aphaPairsChanged)
            iter += 1
        #遍历非边界值
        else:
            nonBoundIs = np.nonzero((oS.alphas.A > 0)*(oS.alphas.A < C))[0]
            for i in nonBoundIs:
                aphaPairsChanged += innerL(i, oS)
                print "non-bound, iter:  %d i,%d, pairs changed %d" %(iter,i,aphaPairsChanged)
                iter += 1
        if entireSet : 
            entireSet = False
        elif (aphaPairsChanged == 0): 
                entireSet = True
        print "iteration number: %d " % iter
    return oS.b, oS.alphas
        
#==============================================================================
# def calcWs(alphas, dataArr, classLabels):
#     dataArr,labelArr = svm.loadDataSet(fileName)
#     X = np.mat(dataArr); labelMat = np.mat(classLabels).transpose()
#     m, n = np.shape(X)
#     w = np.zeros((n, 1))
#     for i in range(m):
#         w += np.multiply(alphas[i]*labelMat[i], X[i, :].T)
#     return w
#==============================================================================
    
#==============================================================================
#  核转换函数
   
def kernelTrans(X,A,kTup):
    m, n = np.shape(X)
    K = np.mat(np.zeros((m,1)))
    if kTup[0] == 'lin': 
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow * deltaRow.T
        K = np.exp(K/(-1*kTup[1]**2))
    else:
        raise NameError("Houseton We Have a Problem -- That Kernel is not recongnized")
    return K

#==============================================================================
# class optStruct:
#     def __init__(self,dataMatIn, classLabels, C, toler, kTup):
#         self.X = dataMatIn
#         self.labelMat = classLabels
#         self.C = C
#         self.tol = toler
#         self.m = np.shape(dataMatIn)[0]
#         self.alphas = np.mat(np.zeros(self.m, 1))
#         self.b = 0
#         self.eCache = np.mat(np.zeros(self.m, 2))
#         self.K = np.mat(np.zeros((self.m, self.m)))
#         for i in range(self.m):
#             self.K[: , i] = kernelTrans(self.X, self.X[i, :], kTup)
#==============================================================================
            
#==============================================================================
# 使用核函数时需要对innerL() 和 calcEk() 函数进行修改
#==============================================================================
# def calcEk(oS,k):
#     fXk = float(np.multiply(oS.alphas, oS.labelMat).T * oS.K[:,k] + oS.b)
#     Ek = fXk - float(oS.labelMat[k])
#     return Ek
# 
# def innerL(i, oS):
#     Ei = calcEk(oS, i)
#     if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
#         #第二个alpha选择中的启发式方法
#         j,Ej = selectJ(i,oS,Ei)
#         alphaIold = oS.alphas[i].copy()
#         alphaJold = oS.alphas[j].copy()
#         if (oS.labelMat[i] != oS.labelMat[j]):
#             L = max(0, oS.alphas[j] - oS.alphas[i])
#             H = min(oS.C,oS.C + oS.alphas[j] - oS.alphas[i])
#         else:
#             L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
#             H = min(oS.C, oS.alphas[j] + oS.alphas[i])
#         if L==H: 
#             print "L==H"; return 0
#         eta = 2.0 * oS.K[i,j] * oS.K[i,i] - oS.X[j,j]
#         if eta >= 0:
#             print "eta>=0"; return 0
#         oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
#         oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
#         #更新误差缓存
#         updateEk(oS,j)
#         if (abs(oS.alphas[j] - alphaJold) < 0.00001):
#             print "j not moving enough"; return 0
#         oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
#         #更新误差缓存
#         updateEk(oS,i)
#         b1 = oS.b-Ei-oS.labelMat[i]*(oS.alphas[i] - alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
#         b2 = oS.b-Ej-oS.labelMat[i]*(oS.alphas[i] - alphaIold)*oS.X[i,j] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
#         if (0< oS.alphas[i]) and (oS.C > oS.alphas[i]):
#             oS.b = b1
#         elif (0< oS.alphas[j]) and (oS.C > oS.alphas[j]):
#             oS.b = b2
#         return 1
#     else:
#         return 0
#==============================================================================
            
#==============================================================================
# 利用核函数进行分类的径向基测试函数
def testRbf(k1 = 1.3):
    fileName = r'E:\ML\ML_source_code\mlia\Ch06\testSetRBF.txt'
    dataArr,labelArr = loadDataSet(fileName)
    b,alphas = smoP(dataArr, labelArr,200, 0.0001, 10000, ('rbf' ,k1))
    datMat = np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print "there are %d Surpport Vectors" % np.shape(sVs)[0]
    m, n =np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :],('rbf', k1))
        pridect = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(pridect) != np.sign(labelArr[i]):
            errorCount += 1
    print "the training error rate is: %f" %(float(errorCount)/m)
    
    fileName = r'E:\ML\ML_source_code\mlia\Ch06\testSetRBF2.txt'
    dataArr,labelArr = loadDataSet(fileName)
    errorCount = 0
    datMat = np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
    m, n =np.shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :],('rbf', k1))
        pridect = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(pridect) != np.sign(labelArr[i]):
            errorCount += 1
    print "the test error rate is: %f" %(float(errorCount)/m)

    
#==============================================================================
#准备数据：将图像转换为测试向量
def img2vector(filename):
    #该函数创建1*1024的NumPy数组
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    #循环出文件的前32行，并将每行的头32行存储在NumPy数组熵，最后返回数组
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect


# 基于SVM的手写识别数字识别
def loadImages(dirName):
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr ==9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i, :] =img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels

def testDigits(kTup = ('rbf', 10)):
    dataArr, labelArr = loadImages(r'E:\ML\ML_source_code\mlia\Ch02\digits\trainingDigits')
    b, alphas = smoP(dataArr,labelArr, 200, 0.0001, 10000, kTup )
    datMat = np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]
    
    
    sVs = datMat[svInd]
    labelSV = labelMat[svInd];
    print "there are %d Surpport Vectors" % np.shape(sVs)[0]
    m, n =np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        pridect = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(pridect) != np.sign(labelArr[i]):
            errorCount += 1
    print "the training error rate is: %f" %(float(errorCount)/m)
    
    dataArr,labelArr = loadImages(r'E:\ML\ML_source_code\mlia\Ch02\digits\testDigits')
    errorCount = 0    
    datMat = np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
    m, n =np.shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        pridect = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(pridect) != np.sign(labelArr[i]):
            errorCount += 1
    print "the test error rate is: %f" %(float(errorCount)/m)
    