import os
from NaiveBayes import *
from separate import *

classType = ['彩票','房产','股票','教育','科技','社会','时尚','体育','娱乐']
classNum = 9


def deleteFile():
    for i in range(classNum):
        temp = []
        rootdir = 'D:\大学课程及作业\数字内容安全\实验四'
        rootdir = rootdir + '\\' + classType[i]
        list = os.listdir(rootdir)      #列出文件夹下所有的目录与文件
        start = int(9*len(list)/10)     #取出进行训练的前9/10的训练集
        for i in range(0,start):
            path = os.path.join(rootdir,list[i])
            if os.path.isfile(path):
                os.remove(path)

def classifyTest():
    result = []
    rootdir = 'D:\大学课程及作业\数字内容安全\实验四'
    classScale = []

    basePath = '词典.txt'
    baseList = sortBase(basePath)

    for i in range(classNum):
        temp = [0,0,0,0,0,0,0,0,0]
        rootdir = rootdir + '\\' + classType[i]
        list = os.listdir(rootdir)      #列出文件夹下所有的目录与文件
        fileNum = len(list)
        classScale.append(fileNum)

        for j in range(0,fileNum):
            path = os.path.join(rootdir,list[j])
            if os.path.isfile(path):
                testEntry = seperate(path,baseList)
                index = testingNB(testEntry)
                temp[index] += 1

        print(i)

    for k in range(classNum):
        result.append(temp[k]/classScale[k])

    print(result)

classifyTest()
