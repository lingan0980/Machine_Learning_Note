

# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 15:52:57 2017

@author: Administrator
"""

import matplotlib.pyplot as plt

使用文本注解绘制树节点

#定义文本框和箭头格式
decisionNode = dict(boxstyle = "sawtooth", fc = "0.8")
leafNode =  dict(boxstyle = "round4", fc = "0.8")
arrow_args = dict(arrowstyle = "<-")

#绘制带箭头的注解,createPlot.ax1是一个全局变量
def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeTxt,xy = parentPt,xycoords = "axes fraction",\
    xytext = centerPt,textcoords = "axes fraction",va = "center",\
    ha = "center",bbox = nodeType ,arrowprops = arrow_args)

#创建新图形并清空绘图区，在绘图区绘制决策节点和叶节点
def createPlot():
    fig = plt.figure(1,facecolor = 'white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111,frameon = False)
    plotNode('decisionNodes',(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode('leafNodes',(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()
   
#==============================================================================
#获取叶节点的数目和树的层数
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        #type()函数,测试节点的数据类型是否为字典
        if type(secondDict[key]).__name__ =='dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs 

#计算遍历过程中的遇到判断节点的个数    
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ =='dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        #如果达到子节点，则从递归调用中返回
        if thisDepth > maxDepth: 
            maxDepth = thisDepth
    return maxDepth
    
def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
           {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head':{0: 'no', 1: 'yes'}}, 1: 'no'}}}}]  
    return listOfTrees[i]

#==============================================================================
# plotTree函数

def plotMidText(cntrPt,parentPt,txtString):
    xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid,yMid,txtString)

'''
全局变量plotTree.tatolW存储树的宽度
全局变量plotTree.tatolD存储树的高度
plotTree.xOff和plotTree.yOff追踪已经绘制的节点位置
'''   
def plotTree(myTree,parentPt,nodeTxt):
    #计算宽与高
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = myTree.keys()[0]
    cntrPt = (plotTree.xOff+(1.0+float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff)
    #标记子节点属性值
    plotMidText(cntrPt,parentPt,nodeTxt)
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    secondDict = myTree[firstStr]
    #减少y偏移
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ =='dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt,leafNode)
            plotMidText((plotTree.xOff,plotTree.yOff),cntrPt,str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

#这个是真正的绘制，上边是逻辑的绘制
def createPlot(inTree):
  fig = plt.figure(1, facecolor='white')
  fig.clf()
  axprops = dict(xticks=[], yticks=[])
  createPlot.ax1 = plt.subplot(111, frameon=False)	#no ticks
  plotTree.totalW = float(getNumLeafs(inTree))
  plotTree.totalD = float(getTreeDepth(inTree))
  plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
  plotTree(inTree, (0.5,1.0), '')
  plt.show()