import nltk
import re
from nltk.tokenize import WordPunctTokenizer  

# nltk.download()

def del_tag(strings):
    dr = re.compile(r'<[^>]+>', re.S)
    if type(strings) == type([]):
        strs = []
        for string in strings:
            string = str(string)
            s = dr.sub('', string)
            strs.append(s)
        return strs
    else:
        strings = str(strings)
        s = dr.sub('', strings)
        return s

del_tag(text)
words = WordPunctTokenizer().tokenize(sentence)
