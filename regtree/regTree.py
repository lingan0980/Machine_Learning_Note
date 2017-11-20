# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 09:46:24 2017

@author: Administrator
"""

#==============================================================================
# class treeNode():
#     def __init__(self, feat, val, right, left):
#         featureToSplitOn = feat
#         valueOfSplit = val
#         rightBranch = right
#         leftBranch = left
#         
#==============================================================================
#伪代码
"""
找到最佳的待切分特征：
    如果该节点不能再分，将该节点存为叶节点
    执行二元切分
    在右子树调用createTree() 方法
    在左子树调用createTree() 方法

"""

#==============================================================================
# CART算法实现
from numpy import *
import numpy as np

def loadDataSet(fileName):
    """
    读取tab键分隔符的文件将每行的内容保存成一组浮点数
    """
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine))
        dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet,feature,value):
    """
    参数：数据集合，待切分的特征和该特征的某个值
    在给定特征和特征值，函数通过数组过滤方式将数据切分得到两个子集
    """
    mat0 = dataSet[np.nonzero(dataSet[:,feature] > value)[0],:]
    mat1 = dataSet[np.nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0,mat1

def regLeaf(dataSet):
    return np.mean(dataSet[:,-1])  #负责生产叶节点


def regErr(dataSet):
    """
    误差估计函数，计算目标变量的平均误差,调用均差函数var
    """
    return np.var(dataSet[:,-1]) * np.shape(dataSet)[0]

#==============================================================================
##伪代码    
    """
    对每个特征：
        对每个特征
            将数据集切分成两份
            计算切分的误差
            如果当前误差小于当前最小误差
                那么将当前切分设定为最佳切分并更新最小误差
    返回最佳的切分的特征和阈值
    """
  
##回归树切分函数
def chooseBestSplit(dataSet,leafType=regLeaf,errType=regErr, ops = (1,4)):
    """
    构建回归树的核心函数，找到最佳的二元切分方式
    tolS 容许的误差下降值
    tolN 切分的最少样本数
    leafType 是对创建叶节点的函数的引用
    errType 是对总方差的计算函数的引用
    ops 是一个用户自定义的参数构成的元组，用以完成树的构建
    """
    tolS = ops[0]; tolN = ops[1]
    #如果剩余特征数为1，停止切分1。
    if len(set(dataSet[:,-1].T.tolist()[0])) ==1:
        return None,leafType(dataSet)   
        # 找不到好的切分特征，调用regLeaf直接生成叶结点
    m, n = np.shape(dataSet)
    S = errType(dataSet)
    bestS = inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set((dataSet[:,featIndex].T.A.tolist())[0]):
            mat0,mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:
        return None, leafType(dataSet)  #如果误差不大则退出
    mat0,mat1 = binSplitDataSet(dataSet,bestIndex,bestValue)
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0]) < tolN:
        return None, leafType(dataSet)  #如果切分的数据集很小则退出
    return bestIndex, bestValue

 
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    """
    leafType=regLeaf 建立节点的函数
    errType = regErr  代表计算误差的函数
    ops=(1,4)  包含树所构建所需其他参数的元组
    """
    feat,val = chooseBestSplit(dataSet,leafType,errType,ops)
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet,feat,val)
    retTree['left'] = createTree(lSet,leafType,errType,ops)
    retTree['right'] = createTree(rSet,leafType,errType,ops)
    return retTree


#==============================================================================
# import matplotlib.pyplot as plt
# myDat = regTree.loadDataSet(r'E:\ML\ML_source_code\mlia\Ch09\ex00.txt')
# myMat = np.mat(myDat)
# regTree.createTree(myMat)
# plt.plot(myMat[:,0],myMat[:,1],'ro') 
# plt.show()         
# 
# myDat1 = regTree.loadDataSet(r'E:\ML\ML_source_code\mlia\Ch09\ex0.txt')
# myMat1 = np.mat(myDat1)
# regTree.createTree(myMat1)
# plt.plot(myMat1[:,1],myMat1[:,2],'ro') 
# plt.show()      
# 
# myDat2 = regTree.loadDataSet(r'E:\ML\ML_source_code\mlia\Ch09\ex2.txt')
# myMat2 = np.mat(myDat2)
# regTree.createTree(myMat2)
# plt.plot(myMat2[:,0],myMat2[:,1],'ro') 
# plt.show()                
#==============================================================================
          
       
#==============================================================================
#函数prue()伪代码：
"""
基于已有的树切分测试数据:
    如果存下任一子集是一棵树，则在该子集递归剪枝过程
    计算将当前两个叶节点合并后的误差
    计算不合并的误差
    如果合并会降低误差的话，就将叶节点合并
"""
      
# 回归树剪枝函数
def isTree(obj):
    '''判断是否为一棵树'''
    return (type(obj).__name__ =='dict')

def getMean(tree):
    '''返回树的平均值'''
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['right'])
    return (tree['left'] + tree['right'])/2.0

#树的后剪枝函数
def prune(tree, testData):
    """
    tree:待剪枝的树
    testData:剪枝所需要的测试数据
    """
    if np.shape(testData)[0] == 0: 
        return getMean(tree) #确认数据集是否为空
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData,tree['spInd'], tree['spVal'])
        
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'],rSet)
        
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(np.power(lSet[:, -1] - tree['left'], 2)) + \
                       sum(np.power(rSet[:, -1] - tree['right'],2)) 
        treeMean = (tree['left'] + tree['right'])/2.0
        errorMerge = sum(np.power(testData[:, -1] - treeMean, 2)) 
        if errorMerge < errorNoMerge:
            print "merging"
            return treeMean
        else:
            return tree
    else:
        return tree
    
#==============================================================================
##模型树的叶节点生成函数
def linearSolve(dataSet):
    m, n = np.shape(dataSet)
    X = np.mat(ones((m,n))); Y = np.mat(ones((m,1)))
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:, -1]
    xTx = X.T * X
    if np.linalg.det(xTx) == 0.0: #判断矩阵是否可逆
        raise NameError('This matrix is singular, connot do inverse, \n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws,X,Y

def moudleLeaf(dataSet):
    ws, X, Y = linearSolve(dataSet)
    return ws

def moudleErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(np.power(Y - yHat, 2))

#==============================================================================
# myMat2 = np.mat(regTree.loadDataSet(r'E:\ML\ML_source_code\mlia\Ch09\exp2.txt'))
# regTree.createTree(myMat2, regTree.moudleLeaf, regTree.moudleErr, (1,10))
# plt.plot(myMat2[:,0],myMat2[:,1],'ro') 
# plt.show()                
#==============================================================================

# 用树回归进行预测的代码
def regTreeEval(model, inDat):
    '''回归树'''
    return float(model)

def modelTreeEval(model, inDat):
    '''模型树'''
    n = np.shape(inDat)[1]
    X = np.mat(ones((1, n+1)))
    X[:,1:n+1] = inDat
    return float(X * model)

def treeForeCast(tree, inData, modelEval = regTreeEval):
    '''对于输入的单个数据点或者行向量，会返回一个浮点值'''
    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        
        if  isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if  isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)
        
def createForeCast(tree, testData, modelEval = regTreeEval):
    '''对数据进行树结构建模'''
    m = len(testData)
    yHat = np.mat(np.zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, np.mat(testData[i]), modelEval)
    return yHat
    

#==============================================================================
# #创建回归树
# trainMat= np.mat(regTree.loadDataSet(r'E:\ML\ML_source_code\mlia\Ch09\bikeSpeedVsIq_train.txt'))
# testMat= np.mat(regTree.loadDataSet(r'E:\ML\ML_source_code\mlia\Ch09\bikeSpeedVsIq_test.txt'))
# myTree = regTree.createTree(trainMat,ops = (1,20))
# yHat = regTree.createForeCast(myTree,testMat[:,0])
# corrcoef(yHat, testMat[:,1], rowvar=0)[0, 1]
#  
# 
# #创建模型树
# myTree = regTree.createTree(trainMat,regTree.moudleLeaf, regTree.moudleErr, ops = (1,20))
# yHat = regTree.createForeCast(myTree, testMat[:,0], regTree.modelTreeEval)
# corrcoef(yHat, testMat[:,1], rowvar=0)[0, 1]
#    
# 
# plt.plot(trainMat[:,0],trainMat[:,1],'ro') 
# plt.show()                
# 
# ws, X, Y = regTree.linearSolve(trainMat) 
# for i in range(np.shape(testMat)[0]):
#     yHat[i] = testMat[i,0]*ws[1,0] + ws[0,0]
# corrcoef(yHat, testMat[:,1], rowvar=0)[0, 1]
#==============================================================================
    