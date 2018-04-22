maxLen = 5 #分词的最大长度
punc = '!#$%^&*+-,./;:<=>?@[\\]_~`|~！·%￥#@……&*（）{}【】|、：；“‘”’？《》，。/'    #待替换的标点符号

#预先处理文件中的标点符号，并以列表形式返回
def getText(fpath):
    f = open(fpath,'r')
    txt = f.read()
    for ch in punc:                 #在txt中遍历punc并进行相关替换
        txt = txt.replace(ch,'\n')
    words = txt.split()             #将字符串转换成列表
    #print(len(words))
    f.close()
    return words

#将生成的列表写入文件
def wtText(fpath,lst):
    f = open(fpath,'a')
    for word in lst:
        item = word + ' '
        f.write(item)
    f.close()

def match(s1,lst):      #字符串匹配
    n = len(s1)
    for i in range(n-1):
        for word in lst:
            temp = s1[-(n-i):]
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

def main():
    rpath = '1-18631.txt'
    wpath = 'text.txt'
    basePath = '词典.txt'

    f =open(basePath,'r')
    txt = f.read()
    base = txt.split()
    f.close()

    words = getText(rpath)
    wordList = divWords(words,base)

    wtText(wpath,wordList)

main()
