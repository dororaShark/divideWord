# coding=UTF-8
import os
import matplotlib.pyplot as plt
import time
import math
import re
from numpy import *

maxLen = 7 #分词的最大长度
punc = '!#$%^&*+-,./;():<=>?@[\\]_~`|~！·%￥#@……&*（）{}【】|、：；“‘”’？《》，。/1234567890qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM'    #待替换的标点符号
classType = ['彩票','房产','股票','教育','科技','社会','时尚','体育','娱乐']
classNum = 9
n = 9

#------------------------------------------------------------------------------
#预先处理文件中的标点符号，并以列表形式返回
def getText(fpath):
    f = open(fpath,'r')
    txt = f.read()
    for ch in punc:                 #在txt中遍历punc并进行相关替换
        txt = txt.replace(ch,' ')
    words = txt.split()             #将字符串转换成列表
    f.close()
    return words

def getText2(fpath):
    f = open(fpath,'r',encoding = 'utf-8',errors='ignore')
    txt = f.read()
    for ch in punc:                 #在txt中遍历punc并进行相关替换
        txt = txt.replace(ch,' ')
    words = txt.split()             #将字符串转换成列表
    f.close()
    return words

def sortBase(basePath): #将词典进行分类
    words=[]
    lst = []

    f =open(basePath,'r')
    txt = f.read()
    base = txt.split()
    f.close()

    for i in range(maxLen-1):
        words.append('\n')
        lst.append('')

    for word in base:
        length = len(word)
        if length <= maxLen:
            words[length-2] = words[length-2] + " " +word

    for j in range(maxLen-1):
        lst[j] = words[j].split()
    return lst

def match(s1,lst):      #字符串匹配
    n = len(s1)
    for i in range(n-1):
        temp = s1[-(n-i):]
        for word in lst[n-2-i]:
            if  temp == word:
                return word,(n-i)
    return s1[-1],1

def divWords(lst,base):
    wordList = []
    for words in lst:
        s2 = ""
        s1 = ""
        if len(words) < maxLen:    #待匹配字符串小于最大分词长度
            s1 = words
            length = len(words)
        else:
            s1 = words[-maxLen:]
            length = maxLen

        while s1:
            subString,n = match(s1,base)
            s2 = s2 + " " + subString

            words = words[:-n]
            if len(words) < maxLen:    #待匹配字符串小于最大分词长度
                s1 = words
                length = len(words)
            else:
                s1 = words[-maxLen:]
                length = maxLen

        s2 = s2.split()
        s2.reverse()
        for item in s2:
            wordList.append(item)
    return wordList

def seperate(rpath,baseList):
    words = getText(rpath)
    wordList = divWords(words,baseList)
    return wordList
#-------------------------------------------------------------------------------------------

def loadTrainDataSet(): #读取训练集
    postingList=[]   #邮件表，二维数组
    classVec=[]
    fileName = ['彩票.txt','房产.txt','股票.txt','教育.txt','科技.txt','社会.txt','时尚.txt','体育.txt','娱乐.txt']
    for i in range(n):
        temp = []
        classVec.append(i)
        temp = getText(fileName[i])
        postingList.append(temp)
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
        pDenom.append(2.0)             #九类样本的总词数

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
        pVect.append(log(pNum[i]/pDenom[i]))    #概率向量(p(x0=1|y=i),p(x1=1|y=i),...p(xn=1|y=i))
        #pVect.append(pNum[i]/pDenom[i])
    return pVect,pAbusive

def classifyNB(vocabList,testArr,pVec,pClass1):  #朴素贝叶斯分类
    testVec=array(setOfWords2Vec(vocabList,testArr))
    p = []

    #此处的乘法并非矩阵乘法，而是矩阵相同位置的2个数分别相乘
    #矩阵乘法应当 dot(A,B) 或者 A.dot(B)
    #下式子是原式子取对数，因此原本的连乘变为连加

    for i in range(9):
        p.append(sum(testVec*pVec[i]) + log(pClass1[i]))


    for u in range(9):
        print(p[u])
    pMax = max(p)

    for j in range(9):
        if p[j] == pMax:
            return j

#测试方法
def testingNB():

    postingList, classVec = loadTrainDataSet()
    vocabList = getText('总词表.txt')
    trainMatrix = createTrainMatrix(vocabList,postingList)
    pVec, pAb = trainNB0(trainMatrix,classVec)

    result = []
    classScale = []

    basePath = '词典.txt'
    baseList = sortBase(basePath)

    result = [0,0,0,0,0,0,0,0,0]

    for i in range(classNum):
        temp = [0,0,0,0,0,0,0,0,0]
        rootdir = classType[i]
        list = os.listdir(rootdir)      #列出文件夹下所有的目录与文件
        fileNum = len(list)
        classScale.append(fileNum)

        for j in range(0,5):
            path = os.path.join(rootdir,list[j])
            if os.path.isfile(path):
                words = getText2(path)
                wordList = divWords(words,baseList)
                judge = classifyNB(vocabList,wordList,pVec,pAb)
                temp[judge] += 1
                print(judge)

        print(temp)
        for k in range(n):
            result[k] = result[k] + temp[k]
        print(result)
        print('----------------------'+str(i)+'------------------')


    print(result)


testingNB()
