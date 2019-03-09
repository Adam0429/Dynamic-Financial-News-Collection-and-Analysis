from senticnet.senticnet import SenticNet
import nltk
from tqdm import tqdm
import xlrd
import pandas as pd
import random
from textblob import TextBlob
import numpy
import IPython
import re
from nltk.corpus import stopwords
import json

def review_to_words(review_text):   
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    words = letters_only.lower().split()                             

    _stopwords = set(stopwords.words("english"))
    _stopwords = nltk.corpus.stopwords.words('english')
    _stopwords.append('would')
    _stopwords.append('kmh')
    _stopwords.append('mph')
    _stopwords.append('  ')
    _stopwords.append('Reuters')                  

    meaningful_words = [w for w in words if not w in _stopwords]   

    return(meaningful_words)

sn = SenticNet()
# concept_info = sn.concept('love')
# polarity_value = sn.polarity_value('love')
# polarity_intense = sn.polarity_intense('love')
# moodtags = sn.moodtags('love')
# semantics = sn.semantics('love')
# sentics = sn.sentics('love') 
 
def sentiment_score(text):
    t = TextBlob(text)
    score = t.sentiment.polarity
    return score
    # tokens = nltk.word_tokenize(text)
    # pos_tags = nltk.pos_tag(tokens)
    # score = 0
    # count = 0
    # for word,tag in pos_tags:
    #   if word in sn.data.keys():
    #      score += float(sn.polarity_intense(word))
    #      count += 1
    #      # print(word,sn.polarity_intense(word))
    # if count == 0: #mid
    #   return -1 
    # return score/count
    
# test sentiment_score accuracy
# scores = []
# workbook = pd.read_csv(u'sentiment.csv',encoding='ISO-8859-1')
# correct = 0
# pos_count = 0
# neg_count = 0
# pos_correct = 0
# neg_correct = 0

# for i in tqdm(range(0,10000)):
#   i = int(random.random()*1599999)
#   if workbook.loc[i][0] == 0:
#      neg_count += 1
#      if workbook.loc[i][0] == sentiment_score_list(workbook.loc[i][5]):
#        neg_correct += 1
#   elif workbook.loc[i][0] == 4:
#      pos_count += 1
#      if workbook.loc[i][0] == sentiment_score_list(workbook.loc[i][5]):
#        pos_correct += 1
# print('pos',pos_correct/pos_count,pos_count,pos_correct)
# print('neg',neg_correct/neg_count,neg_count,neg_correct)


path = r'data/data_labeled.xls'

workbook = xlrd.open_workbook(path)
worksheet = workbook.sheet_by_index(0)
contents = worksheet.col_values(1)
prices = worksheet.col_values(3)

score_list = []
label_list = []

pos_words = {}
neg_words = {}

rates = []
for i in tqdm(range(0,len(contents))):
    tokens = review_to_words(contents[i])
    price_list = json.loads(prices[i])
    price_list = [int(price) for price in price_list]
    rate = []
    for idx in range(1,6):
        rate.append((price_list[idx]-price_list[0])/price_list[0])
    rates.append(rate)
    score_list.append(sentiment_score(contents[i]))
        # if set(list(labels[i])) == {'1'}:
        #     score_list.append(sentiment_score(contents[i]))
        #     label_list.append(1)
        #     for token in tokens:
        #         if token in pos_words.keys():
        #             pos_words[token] += 1
        #         else:
        #             pos_words[token] = 1 
        # if set(list(labels[i])) == {'0'}:
        #     score_list.append(sentiment_score(contents[i]))
        #     label_list.append(0)
        #     for token in tokens:
        #         if token in neg_words.keys():
        #             neg_words[token] += 1
        #         else:
        #             neg_words[token] = 1 

# fiveday_rate_list = []
for i in range(0,5):
    rate = [x[i] for x in rates]
    data = {
           'scores':score_list,
           'rates':rate
           }

    df = pd.DataFrame(data)
    # print(df)
    print(df.corr("kendall"))

# res = sorted(neg_words.items(),key=lambda neg_words:neg_words[1],reverse=False)
# print(res)

# pearson：Pearson相关系数来衡量两个数据集合是否在一条线上面，即针对线性数据的相关系数计算，针对非线性数据便会有误差。
# kendall：用于反映分类变量相关性的指标，即针对无序序列的相关系数，非正太分布的数据
# spearman：非线性的，非正太分析的数据的相关系数


# sheet = workbook.sheet_by_index(1)
# labels = sheet.col_values(0)
# contents = sheet.col_values(5)
# for i in tqdm(range(0,contents)):
#   if labels[i] == sentiment_score_list(contents[1]):
#      correct += 1

    # print(content,sentiment_score_list(content))

# newlist =[i for i in scores if i>0.3]
# print(len(newlist))
# print(correct/len(labels))

# score = sentiment_score_list('i love you very much')  
# print(score)