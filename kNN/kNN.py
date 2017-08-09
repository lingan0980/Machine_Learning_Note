# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 20:32:22 2017

@author: Administrator
"""

import operator
from numpy import *
from os import listdir

def createDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

#==============================================================================
# K近邻算法
def classify0(inX,dataSet,labels,k):
    """应用KNN方法对测试点进行分类，返回一个结果类型 
     
    Keyword argument: 
    testData: 待测试点，格式为数组 
    dataSet： 训练样本集合，格式为矩阵 
    labels： 训练样本类型集合，格式为数组 
    k： 近邻点数 
    """  
    #读取矩阵长度
    dataSetSize=dataSet.shape[0]
    #距离计算
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    #argsort函数返回的是数组值从小到大的索引值
    sortedDistIndicies = distances.argsort() 
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

#==============================================================================
# 将文本记录转换为NumPy的解析程序
def file2matrix(filename):
    fr=open(filename)
    arrayOLines=fr.readlines()
    #得到文本行数
    numberOfLines=len(arrayOLines)
    
    #创建返回的numpy矩阵
    returnMat=zeros((numberOfLines,3))
    classLabelVector = []
    index=0
    
    #解析文本数据到列表
    for line in arrayOLines:
        line=line.strip()    #截取掉所有回车字符
        listFromLine=line.split('\t')
        returnMat[index,:]=listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index +=1
    return returnMat,classLabelVector

#==============================================================================
#归一化特征值        
def autoNorm(dataSet):
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    normDataSet=zeros(shape(dataSet))
    m=dataSet.shape[0]
    normDataSet=dataSet-tile(minVals,(m,1))
    #特征值相除
    normDataSet=normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

#==============================================================================
# 分类器针对约会网站的测试代码
def datingClassTest():
    hoRatio=0.10
    datingDataMat,datingLabels=file2matrix(r'E:\ML\ML_source_code\mlia\Ch02\datingTestSet2.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    m=normMat.shape[0]
    numTestVecs=int(m*hoRatio)
    errorCount=0.0
    for i in range(numTestVecs):
        classifierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],\
                                   datingLabels[numTestVecs:m],3)
        print "the classifier came back with:%d,the real anwer is:%d"\
                                   % (classifierResult,datingLabels[i])
        if (classifierResult!=datingLabels[i]):errorCount +=1.0
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))

#==============================================================================
# 约会网站预测函数
def classifyPerson():
    resultList=['not at all','in small dose','in large doses']
    percentTats=float(raw_input("percentage of time spent playing video games?"))    
    ffMiles=float(raw_input("frequent flier miles earned per year?"))
    iceCream=float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    inArr=array([ffMiles,percentTats,iceCream])
    classifierResult=classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print "You will probably like this person:",resultList[classifierResult -1]
    
#==============================================================================
#准备数据：将图像转换为测试向量
def img2vector(filename):
    #该函数创建1*1024的NumPy数组
    returnVect = zeros((1,1024))
    fr = open(filename)
    #循环出文件的前32行，并将每行的头32行存储在NumPy数组熵，最后返回数组
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

#==============================================================================
#手写数字识别系统的测试代码   
def handwritingClassTest():
    hwLabels = []
    #获取目录内容
    trainingFileList = listdir(r'E:\ML\ML_source_code\mlia\Ch02\digits\trainingDigits')
    #trainingFileList下面有1934个文件
    m = len(trainingFileList)
    #形成了一个1934*1024的0矩阵
    trainingMat = zeros((m,1024))
    #从文件名解分类数字
    for i in range(m):
        #构造要打开的文件名
        fileNameStr = trainingFileList[i]
        #按照"."分开取第一个数
        fileStr = fileNameStr.split('.')[0]
        #按照"_"来分开来取第一数值并强制转换为int类型
        classNumstr = int(fileStr.split('_')[0])
        hwLabels.append(classNumstr)
        trainingMat[i,:] = img2vector(r'E:\ML\ML_source_code\mlia\Ch02\digits\trainingDigits/%s' %fileNameStr)
    testFileList = listdir(r'E:\ML\ML_source_code\mlia\Ch02\digits\testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumstr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector(r'E:\ML\ML_source_code\mlia\Ch02\digits\trainingDigits/%s' %fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print "the classifier came back with: %d, the read answer is:%d" %(classifierResult,classNumstr) 
        #计算错误率
        if (classifierResult !=classNumstr):
            errorCount += 1.0
    print "\nthe total number of errors is %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))
    
