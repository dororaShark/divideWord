# coding=UTF-8
from numpy import *
import matplotlib.pyplot as plt
import time
import math
import re

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
    numTrainDocs = len(trainMatrix)     #样本数量
    numWords = len(trainMatrix[0])      #样本特征数
    pAbusive = sum(trainCategory)/float(numTrainDocs)   #p(y=1)

    #分子赋值为1，分母赋值为2（拉普拉斯平滑）
    p0Num=ones(numWords)   #初始化向量，代表所有彩票类样本中词j出现次数
    p1Num=ones(numWords)   #初始化向量，代表所有房产类样本中词j出现次数
    p2Num=ones(numWords)  #初始化向量，代表所有股票类样本中词j出现次数
    p3Num=ones(numWords)  #初始化向量，代表所有教育类样本中词j出现次数
    p4Num=ones(numWords)  #初始化向量，代表所有科技类样本中词j出现次数
    p5Num=ones(numWords)  #初始化向量，代表所有社会类样本中词j出现次数
    p6Num=ones(numWords)   #初始化向量，代表所有时尚类样本中词j出现次数
    p7Num=ones(numWords)   #初始化向量，代表所有体育类样本中词j出现次数
    p8Num=ones(numWords)   #初始化向量，代表所有娱乐类样本中词j出现次数

    p0Denom=p1Denom=p2Denom=p3Denom=p4Denom=p5Denom=p6Denom=p7Denom=p8Denom=2.0     #九类样本的总词数

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num+=trainMatrix[i]
            p1Denom+=sum(trainMatrix[i])
        elif trainCategory[i] == 0:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
        elif trainCategory[i] == 2:
            p2Num+=trainMatrix[i]
            p2Denom+=sum(trainMatrix[i])
        elif trainCategory[i] == 3:
            p3Num+=trainMatrix[i]
            p3Denom+=sum(trainMatrix[i])
        elif trainCategory[i] == 4:
            p4Num+=trainMatrix[i]
            p4Denom+=sum(trainMatrix[i])
        elif trainCategory[i] == 5:
            p5Num+=trainMatrix[i]
            p5Denom+=sum(trainMatrix[i])
        elif trainCategory[i] == 6:
            p6Num+=trainMatrix[i]
            p6Denom+=sum(trainMatrix[i])
        elif trainCategory[i] == 7:
            p7Num+=trainMatrix[i]
            p7Denom+=sum(trainMatrix[i])
        else:
            p8Num+=trainMatrix[i]
            p8Denom+=sum(trainMatrix[i])

    p0Vect = p0Num/p0Denom  #概率向量(p(x0=1|y=0),p(x1=1|y=0),...p(xn=1|y=0))
    p1Vect = p1Num/p1Denom  #概率向量(p(x0=1|y=1),p(x1=1|y=1),...p(xn=1|y=1))
    p2Vect = p2Num/p2Denom  #概率向量(p(x0=1|y=2),p(x1=1|y=2),...p(xn=1|y=2))
    p3Vect = p3Num/p3Denom  #概率向量(p(x0=1|y=3),p(x1=1|y=3),...p(xn=1|y=3))
    p4Vect = p4Num/p4Denom  #概率向量(p(x0=1|y=4),p(x1=1|y=4),...p(xn=1|y=4))
    p5Vect = p5Num/p5Denom  #概率向量(p(x0=1|y=5),p(x1=1|y=5),...p(xn=1|y=5))
    p6Vect = p6Num/p6Denom  #概率向量(p(x0=1|y=6),p(x1=1|y=6),...p(xn=1|y=6))
    p7Vect = p7Num/p7Denom  #概率向量(p(x0=1|y=7),p(x1=1|y=7),...p(xn=1|y=7))
    p8Vect = p8Num/p8Denom  #概率向量(p(x0=1|y=8),p(x1=1|y=8),...p(xn=1|y=8))

    #取对数，之后的乘法就可以改为加法，防止数值下溢损失精度
    p0Vect=log(p0Vect)
    p1Vect=log(p1Vect)
    p2Vect=log(p2Vect)
    p3Vect=log(p3Vect)
    p4Vect=log(p4Vect)
    p5Vect=log(p5Vect)
    p6Vect=log(p6Vect)
    p7Vect=log(p7Vect)
    p8Vect=log(p8Vect)

    return p0Vect,p1Vect,p2Vect,p3Vect,p4Vect,p5Vect,p6Vect,p7Vect,p8Vect,pAbusive

def classifyNB(vocabList,testEntry,p0Vec,p1Vec,p2Vec,p3Vec,p4Vec,p5Vec,p6Vec,p7Vec,p8Vec,pClass1):  #朴素贝叶斯分类
    #先将输入文本处理成特征向量
    regEx = re.compile('\\W*') #正则匹配分割，以字母数字的任何字符为分隔符
    testArr=regEx.split(testEntry)
    testVec=array(setOfWords2Vec(vocabList,testArr))

    #此处的乘法并非矩阵乘法，而是矩阵相同位置的2个数分别相乘
    #矩阵乘法应当 dot(A,B) 或者 A.dot(B)
    #下式子是原式子取对数，因此原本的连乘变为连加
    '''p1=sum(testVec*p1Vec)+log(pClass1)
    p0=sum(testVec*p0Vec)+log(1.0-pClass1)'''

    p0 = sum(testVec*p0Vec)
    p1 = sum(testVec*p1Vec)
    p2 = sum(testVec*p2Vec)
    p3 = sum(testVec*p3Vec)
    p4 = sum(testVec*p4Vec)
    p5 = sum(testVec*p5Vec)
    p6 = sum(testVec*p6Vec)
    p7 = sum(testVec*p7Vec)
    p8 = sum(testVec*p8Vec)

    if p1>p0:
        return 1
    else:
        return 0

#测试方法
def testingNB():
    postingList,classVec=loadTrainDataSet()
    vocabList=createVocabList(postingList)
    trainMatrix=createTrainMatrix(vocabList,postingList)
    p0V,p1V,p2V,p3V,p4V,p5V,p6V,p7V,p8V,pAb=trainNB0(trainMatrix,classVec)

    #输入测试文本，单词必须用空格分开
    testEntry='fuck you bitch!!!'
    print('测试文本为： '+testEntry)
    if classifyNB(vocabList,testEntry,p0V,p1V,p2V,p3V,p4V,p5V,p6V,p7V,p8V,pAb):
        print("--------侮辱性邮件--------")
    else:
        print("--------正常邮件--------")

testingNB()

