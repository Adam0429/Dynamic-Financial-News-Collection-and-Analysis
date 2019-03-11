from senticnet.senticnet import SenticNet
import nltk
from tqdm import tqdm
import xlrd
import pandas as pd
import random
from textblob import TextBlob
import numpy as np
import IPython
import re
from nltk.corpus import stopwords
import json

# Context-aware Sentiment Detection From Ratings  Yichao Lu, Ruihai Dong, Barry Smyth

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

# sn = SenticNet()
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

count = {}
# neg_words = {}

POS = 0
NEG = 0

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
    if 'not' not in tokens:
        if rate[0]>0:
            POS += len(tokens)
            for token in tokens:
                if token in count.keys():
                    count[token]['pos'] += 1
                else:
                    count[token] = {'pos':1,'neg':0} 

        if rate[0]<0:
            NEG += len(tokens)
            for token in tokens:
                if token in count.keys():
                    count[token]['neg'] += 1
                else:
                    count[token] = {'pos':0,'neg':1} 
    else:
        if rate[0]>0:
            POS += len(tokens)
            for token in tokens:
                token = 'not_'+token
                if token in count.keys():
                    count[token]['pos'] += 1
                else:
                    count[token] = {'pos':1,'neg':0} 

        if rate[0]<0:
            NEG += len(tokens)
            for token in tokens:
                token = 'not_'+token
                if token in count.keys():
                    count[token]['neg'] += 1
                else:
                    count[token] = {'pos':0,'neg':1} 

# 情感极性与五天内新闻涨跌比率的相关度
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




# freq
copy = count.copy()
sent_words = [] # PD>15情感值

for word,value in tqdm(copy.items()):
    if value['pos']+value['neg']<10:
        del count[word]
        continue
    pos = value['pos']/POS
    neg = value['neg']/NEG
    
    value['PD'] = (pos+neg)/(pos-neg) # polarity difference
    if abs(value['PD']) > 20:
        sent_words.append(word) 
    count[word]['sent'] = value['PD']*value['PD'] * np.sign(value['PD'])

sent_words.remove('e')
# res = sorted(count.items(),key=lambda count:count[1]['sent'],reverse=False)
res = sorted(count.items(),key=lambda count:count[1]['PD'],reverse=True)

contents = list(set(contents))
sentiment_feature = {}
sf_len = 0
for w in sent_words:
    for i in range(0,len(contents)):
        if w not in contents[i]:
            continue
        score = sentiment_score(contents[i])
        tokens = review_to_words(contents[i])
        for f in tokens:
            if f != w and f not in sent_words:
                sf_len += 1
                if w+'_'+f not in sentiment_feature.keys():
                    sentiment_feature[w+'_'+f] = {'pos':0,'neg':0}
                else:
                    if score > 0.01:
                        sentiment_feature[w+'_'+f]['pos'] += 1
                    if score < -0.01:
                        sentiment_feature[w+'_'+f]['neg'] += 1

copy = sentiment_feature.copy()
avg_sf = sf_len/len(sentiment_feature.keys())

for word,value in tqdm(copy.items()):
    if value['pos']+value['neg']<avg_sf:
        del sentiment_feature[word]
        continue
    pos = value['pos']/POS
    neg = value['neg']/NEG
    
    value['PD'] = (pos+neg)/(pos-neg) # polarity difference
    sentiment_feature[word]['sent'] = value['PD'] * value['PD'] * np.sign(value['PD'])

# print(sentiment_feature)
res = sorted(sentiment_feature.items(),key=lambda sentiment_feature:sentiment_feature[1]['PD'],reverse=False)
# print(res)
for r in res[:10]:
    print(r[0],r[1]['sent'])

print('========================')

for r in res[-10:]:
    print(r[0],r[1]['sent'])



# for sf,value in sentiment_feature.items():
#     if value['pos']+value['neg'] > avg_sf:
#         print(sf,freq)

# print(count.items())
# IPython.embed()

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