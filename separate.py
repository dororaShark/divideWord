maxLen = 7 #分词的最大长度
punc = '!#$%^&*+-,./;:<=>?@[\\]_~`|~！·%￥#@……&*（）{}【】|、：；“‘”’？《》，。/1234567890qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM'    #待替换的标点符号

#预先处理文件中的标点符号，并以列表形式返回
def getText(fpath):
    f = open(fpath,'r')
    txt = f.read()
    for ch in punc:                 #在txt中遍历punc并进行相关替换
        txt = txt.replace(ch,' ')
    words = txt.split()             #将字符串转换成列表
    #print(len(words))
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

def seperate(rpath):
    basePath = '词典.txt'
    baseList = sortBase(basePath)

    words = getText(rpath)
    wordList = divWords(words,baseList)

    return wordList

