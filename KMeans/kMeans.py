# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 22:50:00 2017

@author: Administrator
"""

"""
创建k个点作为起始质心(经常是随机选择)
当任意一个点的簇分配结果发生改变的时候
    对数据集中的每个数据点
        对每个质心
            计算质心与数据点之间的距离
        将数据点分配到距其最近的簇
    对每一个簇，计算簇中所有点的均值并将均值作为质心
"""

## K-均值聚类支持函数
from numpy import *
import numpy as np

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    return dataMat

def distEclub(vecA, vecB):
    '''计算两点之间的欧式距离'''
    return np.sqrt(sum(np.power(vecA - vecB,2)))

def randCent(dataSet, k):
    '''创建初始质心的函数'''
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k ,n))) 
    for j in range(n):                   #构建簇质心
        minJ = min(dataSet[:, j])
        rangeJ =float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)
    return centroids

##K-均值聚类算法
def kMeans(dataSet, k, distMeas = distEclub, createCent = randCent):
    """
    dataSet: 数据集    k：簇的数目
    disMeas:计算距离   createCent ：创建初始质心的函数
    """
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                #寻找最近的质心
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        print centroids
        for cent in range(k):
            #更新质心的位置
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = np.mean(ptsInClust, axis = 0)
    return centroids, clusterAssment
        
        
#==============================================================================
# import matplotlib.pyplot as plt
# datMat = np.mat(kMeans.loadDataSet(r'E:\ML\ML_source_code\mlia\Ch10\testSet.txt'))
# plt.plot(datMat[:,0],datMat[:,1],'ro') 
# myCentroids, clustAssing = kMeans.kMeans(datMat,4)
# 
# ##绘图代码
# f = plt.figure(1)  
# plt.subplot(111)  
# # with legend  
# plt.scatter(myCentroids[:,0],myCentroids[:,1], marker = 'x',color = 'b', s = 50)  
# plt.scatter(datMat[:,0],datMat[:,1], color = 'r', s = 30)  
# plt.legend(loc = 'upper right')
# plt.show()    
#==============================================================================


##二分K-均值算法的伪代码
"""
将所有点看成一个簇
当簇数目小于K时
对于每一个簇
   计算总误差
   在给定的簇上进行K-均值聚类(k = 2)
   计算将该簇一分为二之后的总误差
选择使得误差最小的那个簇进行划分操作
"""

##二分K-均值聚类算法
def biKmeans(dataSet, k ,distMeas = distEclub):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))
    #创建一个初始簇
    centroid0 = np.mean(dataSet, axis = 0).tolist()[0]
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j ,1] = distMeas(np.mat(centroid0), dataSet[j,:])**2
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            #尝试划分每一簇
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:,0].A==i)[0],:]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:, 1])
            sseNotSplit = sum(clusterAssment[np.nonzero(clusterAssment[:,0].A!=i)[0], 1])
            print "sseSplit, and notSplit:",sseSplit, sseNotSplit
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
                #更新簇的分配结果
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print 'the bestCentToList is:', bestCentToSplit
        print 'the len of bestClustAss is:',len(bestClustAss)
        centList[bestCentToSplit] = bestNewCents[0, :]
        centList.append(bestNewCents[1,:])
        clusterAssment[np.nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:] = bestClustAss
    #return np.mat(centList), clusterAssment
    return centList, clusterAssment
        
                
### 球面距离计算以及簇绘图函数
from math import *

def distSLC(vecA, vecB):
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * cos(pi*(vecB[0,0]-vecA[0,0])/180)
    return arccos(a + b)*6371.0

import matplotlib
import matplotlib.pyplot as plt

def clusterClubs(numClust = 5):
    datList = []
    for line in open(r'E:\ML\ML_source_code\mlia\Ch10\places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
        datMat = np.mat(datList)
        myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas = distSLC)
        
        #读取地图并绘图
        fig = plt.figure()
        rect = [0.1 ,0.1, 0.8, 0.8]
        scatterMarkers = ['s','o','^','8','p','d','v','h','>','<']
        axprops = dict(xticks = [], yticks = [])
        ax0 = fig.add_axes(rect, label = 'ax0', **axprops)
        imgP = plt.imread(r'E:\ML\ML_source_code\mlia\Ch10\Portland.png')
        ax0.imshow(imgP)
        ax1 = fig.add_axes(rect, label = "ax1", frameon = False)
        for i in range(numClust):
            ptsInCurrCluster = datMat[np.nonzero(clustAssing[:,0].A == i)[0],:]
            markerStyle = scatterMarkers[i % len(scatterMarkers)]
            ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0],\
                        ptsInCurrCluster[:, 1].flatten().A[0],\
                        marker = markerStyle, s = 90)
        ax1.scatter(myCentroids[:, 0].flatten().A[0],\
                    myCentroids[:, 1].flatten().A[0],marker = '+', s = 300)
        plt.show()
            
    
    
    
    
 
    
    

      