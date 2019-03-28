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
import inflection as inf
from wordcloud import WordCloud
import enchant
# import matplotlib.pyplot as plt

adj = ['JJ','JJR','JJS']
adv = ['RB','RBR','RBS']
vb = ['VB','VBD','VBG','VBN','VBP','VBZ']
nn = ['NN','NNS']


def review_to_words(review_text): 
    if '(Reuters) -' in review_text:
        review_text = review_text.split('(Reuters) -')[1]
    if '*' in review_text:
        review_text = review_text.split('*')[1]
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    words = letters_only.split()                             
    _stopwords = set(stopwords.words("english"))
    _stopwords = nltk.corpus.stopwords.words('english')
    _stopwords.append('would')
    _stopwords.append('kmh')
    _stopwords.append('mph')
    _stopwords.append('  ')
    _stopwords.append('Reuters')
    _stopwords.append('reuters')                  
    # _stopwords = []
    meaningful_words = [w for w in words if not w in _stopwords]
    return meaningful_words

# d = enchant.Dict("en_US")
import spacy
nlp = spacy.load('en')
def stem_and_check(word):
    word = nlp(word)
    return word[0].lemma_
    # word = inf.singularize(word)
    # word = nltk.PorterStemmer().stem(word)
    # if d.check(word):
    #    return word
    # suggest_words = d.suggest(word)
    # if len(suggest_words) == 0:
    #    return word
    # return suggest_words[0]

def my_read(path):
    file = open(path)
    words = []
    for line in file.readlines():
        words.append(line.strip())
    return words

def output_cloud(count,name):
    # 云图
    text = '' 
    for key,value in count.items():
        text += (key+' ') * (value)
    wc = WordCloud(
        width=1000,
        height=600,
        max_font_size=100,      #字体大小
        min_font_size=10,
        collocations=False, 
        max_words=1000
    )
    wc.generate(text)
    wc.to_file(name+'.png') #图片保存

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
    #    score += float(sn.polarity_intense(word))
    #    count += 1
    #    # print(word,sn.polarity_intense(word))
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
#    neg_count += 1
#    if workbook.loc[i][0] == sentiment_score_list(workbook.loc[i][5]):
#     neg_correct += 1
#   elif workbook.loc[i][0] == 4:
#    pos_count += 1
#    if workbook.loc[i][0] == sentiment_score_list(workbook.loc[i][5]):
#     pos_correct += 1
# print('pos',pos_correct/pos_count,pos_count,pos_correct)
# print('neg',neg_correct/neg_count,neg_count,neg_correct)


path = r'data/labeled_data.xls'

workbook = xlrd.open_workbook(path)
worksheet = workbook.sheet_by_index(0)
_contents = worksheet.col_values(1)
prices = worksheet.col_values(3)

# rates = []
# for i in tqdm(range(0,len(_contents))):
#    rate = []
#    price_list = json.loads(prices[i])
#    for idx in range(0,6):
#       rate.append((price_list[idx+1]-price_list[idx])/price_list[idx])
#    rates.append(rate)
#    score_list.append(sentiment_score(_contents[i]))

# # 情感极性与六天内（包括）新闻涨跌比率的相关度
# # fiveday_rate_list = []
# for i in range(0,6):
#    rate = [x[i] for x in rates]
#    data = {
#         'scores':score_list,
#         'rates':rate
#         }

#    df = pd.DataFrame(data)
#    # print(df)
#    # print(df.corr("kendall"))


datas = []
for i in tqdm(range(0,len(_contents))):
    if '*' not in _contents[i]:
        data = {}
        data['content'] = _contents[i]
        data['tokens'] = review_to_words(data['content'])
        data['tags'] = nltk.pos_tag(data['tokens'])
        price_list = json.loads(prices[i])
        data['rate'] = (price_list[1]-price_list[0])/price_list[0]
        datas.append(data)


score_list = []
label_list = []

count = {}

POS = 0
NEG = 0




for data in tqdm(datas):
    tokens = data['tokens']
    rate = data['rate'] # 选当天的股票变化判断涨跌，因为相关度当天的最高
    if rate>0:
        POS += len(tokens)
        for token in tokens:
            if len(token) < 3:
                continue
            if 'not' in tokens:
                token = 'not_'+token
            if token in count.keys():
                count[token]['pos'] += 1
            else:
                count[token] = {'pos':1,'neg':0} 

    if rate<0:
        NEG += len(tokens)
        for token in tokens:
            if len(token) < 3:
                continue
            if 'not' in tokens:
                token = 'not_'+token
            if token in count.keys():
                count[token]['neg'] += 1
            else:
                count[token] = {'pos':0,'neg':1} 




## freq
copy = count.copy()
sent_words = [] # PD>0.3情感值

# DS sent
for word,value in tqdm(copy.items()):
    if value['pos']+value['neg']<10:
        del count[word]
        continue
    pos = value['pos']/POS
    neg = value['neg']/NEG
    # if value == 'According':
    #    IPython.embed()
    value['PD'] = (pos-neg)/(pos+neg) # polarity difference
    if abs(value['PD']) > 0.3 and nltk.pos_tag([word])[0][1] in adj+adv:  
        sent_words.append(word)
    count[word]['sent'] = value['PD']*value['PD'] * np.sign(value['PD'])
# res = sorted(sent_words.items(),key=lambda sent_words:sent_words[1],reverse=False)
# res = sorted(count.items(),key=lambda count:count[1]['sent'],reverse=False)
# res = sorted(count.items(),key=lambda count:count[1]['PD'],reverse=True)
# print(res)

## neg pos 词
# pos_words = {}
# neg_words = {}
# for word in sent_words:
#    if count[word]['sent'] > 0:
#       pos_words[word.lower()] = count[word]['pos']+count[word]['neg']
#    else:
#       neg_words[word.lower()] = count[word]['pos']+count[word]['neg']

# output_cloud(pos_words,'pos')
# output_cloud(neg_words,'neg')

## 求于bl词典的覆盖率
bl_sent = {}
bl_pos = my_read('sentiment_analysis/bl/positive.txt')  # 4783
bl_neg = my_read('sentiment_analysis/bl/negative.txt')  # 2006


for word in bl_pos:
    bl_sent[word] = 1
for word in bl_pos:
    bl_sent[word] = -1
# pc = 0
# for word in pos_words:
#    if word in bl_pos:
#       pc += 1
# pos_accuracy = pc/len(pos_words)  # 0.2857142857142857

# nc = 0
# for word in neg_words:
#    if word in bl_neg:
#       nc += 1
# neg_accuracy = nc/len(neg_words)  # 0.18



sent_words = [word.lower() for word in sent_words]
feature_words = {}
sentiment_feature = {}

for data in tqdm(datas):
    tokens = data['tokens']
    tags = data['tags']
    for word,tag in tags:
        if tag not in vb+nn or len(word)<3:
            continue
        # word = stem_and_check(word)
        if word not in feature_words.keys():
            feature_words[word] = 1
        else:
            feature_words[word] += 1

avg_f = sum([item[1] for item in feature_words.items()])/len(feature_words.keys())
copy = feature_words.copy()

for word,value in tqdm(copy.items()):
    if value<avg_f:
        del feature_words[word]

feature_words = [inf.singularize(word).lower() for word in feature_words.keys()]

sf_len = 0
for data in tqdm(datas):
    tokens = data['tokens']
    rate = data['rate']
    tokens = [inf.singularize(token).lower() for token in tokens]
    for w in sent_words:
        if w not in tokens:
            continue
        for f in feature_words:
            if f in tokens and f != w:
                if abs(tokens.index(w)-tokens.index(f))<3:
                    sf_len += 1
                    if w+'_'+f not in sentiment_feature.keys():
                        if rate > 0:
                            sentiment_feature[w+'_'+f] = {'pos':1,'neg':0}
                        if rate < 0:
                            sentiment_feature[w+'_'+f] = {'pos':0,'neg':1}
                    else:
                        if rate > 0:
                            sentiment_feature[w+'_'+f]['pos'] += 1
                        if rate < 0:
                            sentiment_feature[w+'_'+f]['neg'] += 1

copy = sentiment_feature.copy()
# avg_sf = sf_len/len(sentiment_feature.keys())


for word,value in tqdm(copy.items()):
    if value['pos']+value['neg']<6: #avg_sf
        try:
            del sentiment_feature[word]
        except:
            pass
        continue
    pos = value['pos']/POS
    neg = value['neg']/NEG
    
    value['PD'] = (pos-neg)/(pos+neg) # polarity difference
    sentiment_feature[word]['sent'] = value['PD'] * value['PD'] * np.sign(value['PD'])

res = sorted(sentiment_feature.items(),key=lambda sentiment_feature:sentiment_feature[1]['sent'],reverse=False)

## company word
# company_pos = {}
# company_neg = {}
# for key,value in sentiment_feature.items():
#    if 'company' in key:
#       if sentiment_feature[key]['sent'] > 0:
#          company_pos[key.split('_')[0]] = sentiment_feature[key]['pos']
#       else:
#          company_neg[key.split('_')[0]] = sentiment_feature[key]['neg']

# output_cloud(company_pos,'company_pos')
# output_cloud(company_neg,'company_neg')


# for r in res[:30]:
#    print(r[0],r[1]['sent'],r[1]['pos']+r[1]['neg'])
#    print(' ')

# print(' ')
# print('========================')

# for r in res[-40:]:
#    print(r[0],r[1]['sent'],r[1]['pos'],r[1]['neg'])
#    print(' ')

# 展示
# pos_res = {}
# for r in res:
#     if r[1]['sent'] == 1.0:
#         pos_res[r[0]] = r[1]['pos']+r[1]['neg']
# pos_res = sorted(pos_res.items(),key=lambda pos_res:pos_res[1],reverse=True)
# for r in pos_res[:20]:
#     print(r[0],'1.0',r[1])
#     print(' ')

# print(' ')
# print('========================')
# print(' ')

# neg_res = {}
# for r in res:
#     if r[1]['sent'] == -1.0:
#         neg_res[r[0]] = r[1]['pos']+r[1]['neg']
# neg_res = sorted(neg_res.items(),key=lambda neg_res:neg_res[1],reverse=True)
# for r in neg_res[:20]:
#     print(r[0],'-1.0',r[1])
#     print(' ')
# # for sf,value in sentiment_feature.items():
#    if value['pos']+value['neg'] > avg_sf:
#       print(sf,freq)

IPython.embed()
# Predict
# content to vector

for data in tqdm(datas):
    idx = datas.index(data)
    tags = data['tags']
    datas[idx]['DsVector'] = [0,0,0,0]
    datas[idx]['SnVector'] = [0,0,0,0]
    datas[idx]['BlVector'] = [0,0,0,0]
    for word,tag in tags:
        if word in bl.keys():
            print(word)
        if tag in adj:
            if word in count.keys():
                datas[idx]['DsVector'][0] += count[word]['sent']
            if word in sn.data.keys():
                datas[idx]['SnVector'][0] += float(sn.polarity_intense(word))
            if word in bl_sent.keys():
                datas[idx]['BlVector'][0] += bl_sent[word]
        elif tag in adv:
            if word in count.keys():
                datas[idx]['SnVector'][1] += count[word]['sent']
            if word in sn.data.keys():
                datas[idx]['SnVector'][1] += float(sn.polarity_intense(word))
            if word in bl_sent.keys():
                datas[idx]['BlVector'][0] += bl_sent[word]  
        elif tag in nn:
            if word in count.keys():
                datas[idx]['DsVector'][2] = count[word]['sent']
            if word in sn.data.keys():
                datas[idx]['SnVector'][2] += float(sn.polarity_intense(word))
            if word in bl_sent.keys():
                datas[idx]['BlVector'][0] += bl_sent[word]
        elif tag in vb:
            if word in count.keys():
                datas[idx]['DsVector'][3] = count[word]['sent']
            if word in sn.data.keys():
                datas[idx]['SnVector'][3] += float(sn.polarity_intense(word))
            if word in bl_sent.keys():
                datas[idx]['BlVector'][0] += bl_sent[word]
    # datas[idx]['DsVector'] = [adv_score,adv_score,noun_score,verb_score]

from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.model_selection import KFold

X = [data['DsVector'] for data in datas]
X = [data['SnVector'] for data in datas]
X = [data['BlVector'] for data in datas]
Y = [np.sign(data['rate']) for data in datas]

# x_train,x_test,y_train,y_test = model_selection.train_test_split(X,Y,test_size=0.1,shuffle=True)
# clf = GaussianNB()
# clf.fit(np.array(x_train), np.array(y_train))
# print('准确率：',clf.score(np.array(test_x), np.array(test_y))) 

scores = 0
kf = KFold(n_splits=10,shuffle=True)
for train_index, test_index in kf.split(X):
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for index in train_index:
        train_x.append(X[index])
        train_y.append(Y[index])
    for index in test_index:
        test_x.append(X[index])
        test_y.append(Y[index])

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    clf = GaussianNB()
    clf.fit(X, Y)
    scores += clf.score(test_x, test_y)
    print('准确率：',clf.score(test_x, test_y))  # 计算测试集的度量值（准确率）
print('平均准确率：',scores/10)

# sheet = workbook.sheet_by_index(1)
# labels = sheet.col_values(0)
# contents = sheet.col_values(5)
# for i in tqdm(range(0,contents)):
#   if labels[i] == sentiment_score_list(contents[1]):
#    correct += 1

    # print(content,sentiment_score_list(content))

# newlist =[i for i in scores if i>0.3]
# print(len(newlist))
# print(correct/len(labels))

# score = sentiment_score_list('i love you very much')  
# print(score)