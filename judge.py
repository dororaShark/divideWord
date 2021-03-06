# coding=UTF-8
from numpy import *
from separate import *
import matplotlib.pyplot as plt
import time
import math
import re

n = 9

def loadTrainDataSet(): #读取训练集
    fileIn = open('testSet.txt')
    postingList=[]   #邮件表，二维数组
    classVec=[]
    i=0
    for line in fileIn.readlines():
        lineArr = line.strip().split()
        temp=[]
        for i in range(len(lineArr)):
            if i==0:
                classVec.append(int(lineArr[i]))
            else:
                temp.append(lineArr[i])
        postingList.append(temp)
        i=i+1
    return postingList,classVec


def createVocabList(dataSet):  #创建词典
    vocabSet = set([])  #定义list型的集合
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):  #对于每一个训练样本，得到其特征向量
    returnVec= [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            pass
            #print("\'%s\' 不存在于词典中"%word)
    return returnVec

def createTrainMatrix(vocabList,postingList):  #生成训练矩阵，即每个样本的特征向量
    trainMatrix=[]   #训练矩阵
    for i in range(len(postingList)):
        curVec=setOfWords2Vec(vocabList,postingList[i])
        trainMatrix.append(curVec)
    return trainMatrix


def trainNB0(trainMatrix,trainCategory):
    pAbusive = []
    pNum = []
    pDenom = []
    pVect = []
    fileNum = [5470,6155,6044,5724,5670,5808,5711,5820,6018]    #各个类别训练集数目

    numWords = len(trainMatrix[0])      #样本特征数
    numTrainDocs = len(trainMatrix)     #测试集类别数

    for i in range(9):
        pAbusive.append(fileNum[i]/float(sum(fileNum)))   #p(y=1)

    #分子赋值为1，分母赋值为2（拉普拉斯平滑）
    for j in range(9):
        pNum.append(ones(numWords)) #初始化向量，代表该类样本中词j出现次数
        pDenom.append(2.0)          #九类样本的总词数

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            pNum[1]+=trainMatrix[i]
            pDenom[1]+=sum(trainMatrix[i])
        elif trainCategory[i] == 0:
            pNum[0]+=trainMatrix[i]
            pDenom[0]+=sum(trainMatrix[i])
        elif trainCategory[i] == 2:
            pNum[2]+=trainMatrix[i]
            pDenom[2]+=sum(trainMatrix[i])
        elif trainCategory[i] == 3:
            pNum[3]+=trainMatrix[i]
            pDenom[3]+=sum(trainMatrix[i])
        elif trainCategory[i] == 4:
            pNum[4]+=trainMatrix[i]
            pDenom[4]+=sum(trainMatrix[i])
        elif trainCategory[i] == 5:
            pNum[5]+=trainMatrix[i]
            pDenom[5]+=sum(trainMatrix[i])
        elif trainCategory[i] == 6:
            pNum[6]+=trainMatrix[i]
            pDenom[6]+=sum(trainMatrix[i])
        elif trainCategory[i] == 7:
            pNum[7]+=trainMatrix[i]
            pDenom[7]+=sum(trainMatrix[i])
        else:
            pNum[8]+=trainMatrix[i]
            pDenom[8]+=sum(trainMatrix[i])

    for i in range(9):
        #取对数，之后的乘法就可以改为加法，防止数值下溢损失精度
        #pVect.append(log(pNum[i]/pDenom[i]))    #概率向量(p(x0=1|y=8),p(x1=1|y=8),...p(xn=1|y=8))
        pVect.append(log(pNum[i]/pDenom[i]))
    return pVect,pAbusive

def classifyNB(vocabList,testArr,pVec,pClass1):  #朴素贝叶斯分类
    testVec=array(setOfWords2Vec(vocabList,testArr))
    p = []

    #此处的乘法并非矩阵乘法，而是矩阵相同位置的2个数分别相乘
    #矩阵乘法应当 dot(A,B) 或者 A.dot(B)
    #下式子是原式子取对数，因此原本的连乘变为连加

    for i in range(9):
        p.append(sum(testVec*pVec[i]) + log(pClass1[i]))

    pMax = max(p)

    for j in range(9):
        if p[j] == pMax:
            return j


#测试方法
def testingNB():
    postingList, classVec = loadTrainDataSet()
    vocabList = vocabList=createVocabList(postingList)
    trainMatrix = createTrainMatrix(vocabList,postingList)
    pVec, pAb = trainNB0(trainMatrix,classVec)

    basePath = '词典.txt'
    baseList = sortBase(basePath)

    fpath = 'test.txt'
    testEntry = seperate(fpath,baseList)

    judge = classifyNB(vocabList,testEntry,pVec,pAb)
    if judge == 0:
        print("--------彩票--------")
    elif judge == 1:
        print("--------房产--------")
    elif judge == 2:
        print("--------股票--------")
    elif judge == 3:
        print("--------教育--------")
    elif judge == 4:
        print("--------科技--------")
    elif judge == 5:
        print("--------社会--------")
    elif judge == 6:
        print("--------时尚--------")
    elif judge == 7:
        print("--------体育--------")
    else:
        print("--------娱乐--------")

testingNB()
