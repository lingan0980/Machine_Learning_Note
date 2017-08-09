# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 16:17:56 2017

@author: Administrator
"""

import operator
from math import log

#计算给数据集的香农熵
def calcShannonEnt(dataSet):
    #计算数据集中实例的总数
    numEntries = len(dataSet)
    #创建一个数据字典
    labelCounts = {}
    
    for featVec in dataSet:
        #取键值最后一列的数值的最后一个字符串
        currentLabel = featVec[-1]
        #键值不存在就使当前键加入字典
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        #以2为底求对数
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

def createDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels

#==============================================================================
#dataSet:待划分的数据集
#axis划分数据集的特征
#特征的返回值
def splitDataSet(dataSet,axis,value):
    #创建新的list对象
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
            
#==============================================================================
# 按照给定的特征划分数据集
def choooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) -1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature =-1
    for i in range(numFeatures):
        #创建唯一的分类标签列表
        #取dataSet的第i个数据的第i个数据，并写入列表
        featList = [example[i] for example in dataSet]
        #将列表的数据集合在一起，并去重
        uniqueVals = set(featList)
        newEntropy = 0.0
        #计算每种划分方式的信息熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        #计算好信息熵增益
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

#==============================================================================
# 得出次数最多的分类名称
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] +=1
    sortedClassConnt = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassConnt[0][0]

#==============================================================================
# 创建树的函数代码
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    #类别完全相同则停止继续划分
    if classList.count(classList[0]) ==len(classList):
        return classList[0]
    #遍历完所有的特征时返回出现次数最多的类别
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = choooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    #得到列表包含的所有属性
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet\
                            (dataSet,bestFeat,value),subLabels)
    return myTree

#==============================================================================
#使用决策树的分类函数
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    #将标签字符串转换为索引
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel            
            
#==============================================================================
#使用pickle模块存储决策树
def storeTree(inputTree,filename)        :
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree,fw)
    fw.close
    
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)

    
    
        
    












