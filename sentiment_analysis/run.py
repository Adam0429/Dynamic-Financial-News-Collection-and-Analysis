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
from gensim.models import Word2Vec
import time
from nltk import sent_tokenize
from senticnet.senticnet import SenticNet
from wordcloud import WordCloud
import datetime
# import matplotlib.pyplot as plt
adj = ['JJ','JJR','JJS']
adv = ['RB','RBR','RBS']
vb = ['VB','VBD','VBG','VBN','VBP','VBZ']
nn = ['NN','NNS']

sn = SenticNet()

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
    meaningful_words = [w for w in words if w not in _stopwords]
    return meaningful_words

# d = enchant.Dict("en_US")
# import spacy
# nlp = spacy.load('en')
# def stem_and_check(word):
#     word = nlp(word)
#     return word[0].lemma_
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



def vote(results,datas):
    if len(datas) != len(results):
        raise("vote error")
    vote_dict = {}
    date_set = set([data['date'] for data in datas])
    company_set = set([data['company'] for data in datas])
    for company in company_set:
        vote_dict[company] = {}
        for date in date_set:
            vote_dict[company][date] = set()
    for idx in range(0,len(datas)):
        date = datas[idx]['date']
        company = datas[idx]['company']         
        vote_dict[company][date].add(idx)
        
    new_results = []
    print('voting')
    for i in range(0,len(results)):
        count = 0
        company = datas[i]['company']
        date = datas[i]['date']
        for idx in vote_dict[company][date]:
            count += results[idx]
        if count < 0:  
            new_results.append(np.float64(-1))
        else:
            new_results.append(np.float64(1))
    return np.array(new_results)

def time_fun(date,day):
    st = datetime.datetime.strptime(date, '%Y-%m-%d')
    et = st - datetime.timedelta(days=day)
    return et.strftime("%Y-%m-%d")

def accuracy(y,y2):
    if len(y) != len(y2):
        raise Exception("error")
    count = 0
    for i in range(0,len(y)):
        if y[i] == y2[i]:
            count += 1
    return count/len(y2)

def recall(y,y2):
    if len(y) != len(y2):
        raise Exception("error")
    count = 0
    for i in range(0,len(y)):
        if y[i] == y2[i]:
            count += 1
    return count/len(y2)

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
    
def load_data(path):
    workbook = xlrd.open_workbook(path)
    worksheet = workbook.sheet_by_index(0)
    contents = worksheet.col_values(1)
    companies = worksheet.col_values(2)
    prices = worksheet.col_values(3)
    dates = worksheet.col_values(4)
    datas = []
    for i in tqdm(range(0,len(contents))):
    # if '*' not in contents[i]:
        # if companies[i] != 'Apple Inc.':
        #     continue
        price_list = json.loads(prices[i])   
        if price_list[2] == price_list[1]:
            continue
        data = {}
        data['content'] = contents[i]
        sents = sent_tokenize(data['content'])
        data['tokens'] = []
        data['tags'] = []
        for sent in sents:
            token = review_to_words(sent) # 去停用词会影响词性标注吗？？
            data['tokens'].append(token)
            data['tags'].append(nltk.pos_tag(token))
        data['company'] = companies[i]
        data['rate'] = (price_list[2]-price_list[1])/price_list[1]
        data['date'] = dates[i]
        datas.append(data)

    return datas

def train_sent_dict(datas):
    count = {}

    POS = 0
    NEG = 0

    pos_count = 0
    neg_count = 0

    N = 0 #len of tokens
        
    for data in datas:
        for tokens in data['tokens']: 
            N += len(tokens)
            rate = data['rate'] # 选当天的股票变化判断涨跌，因为相关度当天的最高
            if rate>0:
                pos_count += 1
                POS += len(tokens)
                for token in tokens:
                    if len(token) < 3:
                        continue
                    if 'not' in tokens:
                        token = 'not_'+token
                    if token in count.keys():
                        count[token]['pos'] += 1
                        count[token]['pos_rate'] += rate
                    else:
                        count[token] = {'pos':1,'neg':0,'pos_rate':rate,'neg_rate':0} 

            if rate<0:
                neg_count += 1
                NEG += len(tokens)
                for token in tokens:
                    if len(token) < 3:
                        continue
                    if 'not' in tokens:
                        token = 'not_'+token
                    if token in count.keys():
                        count[token]['neg'] += 1
                        count[token]['neg_rate'] -= rate
                    else:
                        count[token] = {'pos':0,'neg':1,'pos_rate':0,'neg_rate':-rate}
                        ## freq
    copy = count.copy()
    sent_words = [] # PD>0.3情感值

    freq_pos = pos_count/len(datas) 
    freq_neg = neg_count/len(datas)

    # DS sent and PMI sent
    for word,value in copy.items():
        if value['pos']+value['neg']<len(datas)/100:
            del count[word]
            continue
        freq_w_pos = value['pos']/len(datas)
        freq_w_neg = value['neg']/len(datas)
        freq_w = (value['pos']+value['neg'])/len(datas)
        if freq_w_pos*N == 0:
            PMI_w_pos = 0
        else:
            PMI_w_pos = np.log2(freq_w_pos*N/freq_w*freq_pos)
        if freq_w_neg*N == 0:
            PMI_w_neg = 0
        else:
            PMI_w_neg = np.log2(freq_w_neg*N/freq_w*freq_neg)
        count[word]['PMI_sent'] = PMI_w_pos - PMI_w_neg

        pos = value['pos']/len(datas)
        neg = value['neg']/len(datas)
        value['PD'] = (pos-neg)/(pos+neg) # polarity difference
        if abs(value['PD']) > 0.3 and nltk.pos_tag([word])[0][1] in adj+adv:  
            sent_words.append(word)
        count[word]['sent'] = value['PD']*value['PD'] * np.sign(value['PD'])

        pos_rate = value['pos_rate']/len(datas)
        neg_rate = value['neg_rate']/len(datas)
        value['PD_rate'] = (pos_rate-neg_rate)/(pos_rate+neg_rate) # polarity difference
        count[word]['sent_rate'] = value['PD_rate']*value['PD_rate'] * np.sign(value['PD_rate'])
    return count


def news2vector(datas,count,bl_sent):
    for idx in range(0,len(datas)):    
        datas[idx]['DsVector'] = [0,0,0,0]
        datas[idx]['DsVector_rate'] = [0,0,0,0]
        datas[idx]['SnVector'] = [0,0,0,0]
        datas[idx]['BlVector'] = [0,0,0,0]
        datas[idx]['PmiVector'] = [0,0,0,0]
        datas[idx]['ContextVector'] = [0,0,0,0]
        
        
        for tags in datas[idx]['tags']:
            for word,tag in tags:
                if tag in adj:
                    if word in count.keys():
                        datas[idx]['DsVector'][0] += count[word]['sent']
                        datas[idx]['DsVector_rate'][0] += count[word]['sent_rate']
                        datas[idx]['PmiVector'][0] += count[word]['PMI_sent']
                    if word in sn.data.keys():
                        datas[idx]['SnVector'][0] += float(sn.polarity_intense(word))
                    if word in bl_sent.keys():
                        datas[idx]['BlVector'][0] += bl_sent[word]
                elif tag in adv:
                    if word in count.keys():
                        datas[idx]['DsVector'][1] += count[word]['sent']
                        datas[idx]['DsVector_rate'][1] += count[word]['sent_rate']
                        datas[idx]['PmiVector'][1] += count[word]['PMI_sent']
                    if word in sn.data.keys():
                        datas[idx]['SnVector'][1] += float(sn.polarity_intense(word))
                    if word in bl_sent.keys():
                        datas[idx]['BlVector'][1] += bl_sent[word]  
                elif tag in nn:
                    if word in count.keys():
                        datas[idx]['DsVector'][2] = count[word]['sent']
                        datas[idx]['DsVector_rate'][2] += count[word]['sent_rate']
                        datas[idx]['PmiVector'][2] += count[word]['PMI_sent']
                    if word in sn.data.keys():
                        datas[idx]['SnVector'][2] += float(sn.polarity_intense(word))
                    if word in bl_sent.keys():
                        datas[idx]['BlVector'][2] += bl_sent[word]
                elif tag in vb:
                    if word in count.keys():
                        datas[idx]['DsVector'][3] = count[word]['sent']
                        datas[idx]['DsVector_rate'][3] += count[word]['sent_rate']
                        datas[idx]['PmiVector'][3] += count[word]['PMI_sent']
                    if word in sn.data.keys():
                        datas[idx]['SnVector'][3] += float(sn.polarity_intense(word))
                    if word in bl_sent.keys():
                        datas[idx]['BlVector'][3] += bl_sent[word]

def news2vector2(datas,count,bl_sent):
    for idx in range(0,len(datas)):
        datas[idx]['DsVector_rate'] = [0,0,0,0]
        for tags in datas[idx]['tags']:
            for word,tag in tags:
                if tag in adj:
                    if word in count.keys():
                        datas[idx]['DsVector_rate'][0] += count[word]['sent_rate']

                elif tag in adv:
                    if word in count.keys():
                        datas[idx]['DsVector_rate'][1] += count[word]['sent_rate']

                elif tag in nn:
                    if word in count.keys():
                        datas[idx]['DsVector_rate'][2] += count[word]['sent_rate']

                elif tag in vb:
                    if word in count.keys():
                        datas[idx]['DsVector_rate'][3] += count[word]['sent_rate']
        # datas[idx]['DsVector'] = [adv_score,adv_score,noun_score,verb_score]

import pickle
datas = pickle.load(open('/home/stocksentiment/datas.pkl','rb'))
count = pickle.load(open('/home/stocksentiment/sent_dict.pkl', 'rb'))
print('load finish')

path = r'~/Dynamic-Financial-News-Collection-and-Analysis/labeled_data.xls'

# datas = load_data(path)
# count = train_sent_dict(datas)

# import pickle
# output = open('sent_dict.pkl', 'wb')
# input = open('sent_dict.pkl', 'rb')
# s = pickle.dump(count, output)
# output.close()
# clf2 = pickle.load(input)
# input.close()
# print clf2.predict(X[0:1])

## 新闻情感值和股价的相关度
workbook = xlrd.open_workbook(path)
worksheet = workbook.sheet_by_index(0)
contents = worksheet.col_values(1)
companies = worksheet.col_values(2)
prices = worksheet.col_values(3)
dates = worksheet.col_values(4)

rates = []
score_list = []

for i in range(0,len(contents)):
    rate = []
    price_list = json.loads(prices[i])
    for idx in range(0,6):
        rate.append((price_list[idx+1]-price_list[idx])/price_list[idx])
    rates.append(rate)
    score_list.append(sentiment_score(contents[i]))
    
## 合并一天新闻
# for i in tqdm(range(0,len(contents))):
#     rate = []
#     price_list = json.loads(prices[i])
#     for idx in range(0,6):
#       rate.append((price_list[idx+1]-price_list[idx])/price_list[idx])
#     if rate in rates:
#         idx = rates.index(rate)
#         score_list[idx] += sentiment_score(contents[i])
#         continue
#     rates.append(rate)
#     score_list.append(sentiment_score(contents[i]))

# 情感极性与六天内（包括）新闻涨跌比率的相关度

# 股票与新闻情感相关性
fiveday_rate_list = []
for i in range(0,6):
    rate = [x[i] for x in rates]
    data = {
        'scores':score_list,
        'rates':rate
        }

    df = pd.DataFrame(data)
   # print(df)
    print(df.corr("kendall"))

# wordvec
tokens = []
for data in datas:
    for ts in data['tokens']:
        tokens.append(ts)
model = Word2Vec(sentences = tokens,min_count = 10)

from nltk.tokenize import WordPunctTokenizer  
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
import xlrd,xlwt
import numpy as np
# import spacy
from gensim.models import Word2Vec
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical

sentences = [data['content'] for data in datas]

bag_of_keywords = set(['fail', 'success', 'win', 'drop', 'rise', 'shrink', 'jump', 'gain', 'down', 'up'])
stop = False
bok_size = 100
for i in range(10):
    new_words = []
    if stop:break
    for k in bag_of_keywords:
        if k in model.wv.vocab.keys():# wv = wordvector
            new_words.extend(model.most_similar(k))
for n in new_words:
    if n[0].islower() and len(n[0])>3 and n[0].isalpha():
        bag_of_keywords.add(n[0])
        if len(bag_of_keywords) == bok_size:
            stop = True
            break

'''fit():计算数据的参数，\mu（均值），\sigma（标准差），并存储在对象中（例如实例化的CountVectorizer()等）。
transform():将这些参数应用到数据集，进行标准化（尺度化）。'''

## Bag of keywords 统计词语的两个api
bag_of_keywords = np.array(list(bag_of_keywords))
bok_tfidf = TfidfVectorizer(lowercase = False, min_df = 1, vocabulary=bag_of_keywords)
X_bok_tfidf = bok_tfidf.fit_transform(sentences)
X_bok_tfidf = X_bok_tfidf.toarray()

## Category tag
category_tags = set(['published','presented','unveil','investment','bankrupt','acquisition','government'
                     'sue','lawsuit','highlights'])
stop = False
cate_size = 100

for _ in range(10):
    new_words = []
    if stop:break
    for k in category_tags:
        if k in model.wv.vocab.keys():
            new_words.extend(model.most_similar(k))
    for n in new_words:
        if n[0].islower() and len(n[0])>3 and n[0].isalpha():
            category_tags.add(n[0])
            if len(category_tags) == cate_size:
                stop = True
                break


category_tags = np.array(list(category_tags))

ct_tfidf = TfidfVectorizer(lowercase = False, min_df = 1, vocabulary = category_tags)
X_ct_idf = ct_tfidf.fit_transform(sentences)
X_ct_idf = X_ct_idf.toarray()

full_tfidf = TfidfVectorizer(lowercase=False, min_df = 1,vocabulary=bag_of_keywords,use_idf=False)
X_full_tfidf = full_tfidf.fit_transform(sentences)
X_full_tfidf = X_full_tfidf.toarray()

# x_bok RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from keras.utils import to_categorical
import matplotlib.pyplot as plt

print('X_bok_tfidf')
X = X_bok_tfidf
train_x,test_x,train_y,test_y=model_selection.train_test_split(X,Y,test_size=0.2,shuffle=False)
clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
# clf = LinearRegression()
clf.fit(np.array(train_x), np.array(train_y))
predict_y = clf.predict(test_x)
print('accuracy：',clf.score(np.array(test_x), np.array(test_y))) 
print('recall：',recall_score(test_y,clf.predict(test_x),average = 'macro'))
print('precision：',precision_score(test_y, clf.predict(test_x), average='macro'))
    
print('X_ct_idf')
X = X_ct_idf
train_x,test_x,train_y,test_y=model_selection.train_test_split(X,Y,test_size=0.2,shuffle=False)
clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
# clf = LinearRegression()
clf.fit(np.array(train_x), np.array(train_y))
predict_y = clf.predict(test_x)
print('accuracy：',clf.score(np.array(test_x), np.array(test_y))) 
print('recall：',recall_score(test_y,clf.predict(test_x),average = 'macro'))
print('accuracy：',precision_score(test_y, clf.predict(test_x), average='macro'))

print('X_full_tfidf')
X = X_full_tfidf
train_x,test_x,train_y,test_y=model_selection.train_test_split(X,Y,test_size=0.2,shuffle=False)
clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
# clf = LinearRegression()
clf.fit(np.array(train_x), np.array(train_y))
predict_y = clf.predict(test_x)
print('accuracy：',clf.score(np.array(test_x), np.array(test_y))) 
print('recall：',recall_score(test_y,clf.predict(test_x),average = 'macro'))
print('accuracy：',precision_score(test_y, clf.predict(test_x), average='macro'))


# x_bok RandomForestClassifier
print('X_bok_tfidf')
X = X_bok_tfidf
train_x,test_x,train_y,test_y=model_selection.train_test_split(X,Y,test_size=0.2,shuffle=False)
clf = GaussianNB()
# clf = LinearRegression()
clf.fit(np.array(train_x), np.array(train_y))
predict_y = clf.predict(test_x)
print('accuracy：',clf.score(np.array(test_x), np.array(test_y))) 
print('recall：',recall_score(test_y,clf.predict(test_x),average = 'macro'))
print('precision：',precision_score(test_y, clf.predict(test_x), average='macro'))

vote_predict_y = vote(predict_y,datas[-len(test_x):])
vote_recall = recall_score(test_y,vote_predict_y,average = 'macro')
vote_precision = precision_score(test_y, vote_predict_y, average='macro')

    
print('X_ct_idf')
X = X_ct_idf
train_x,test_x,train_y,test_y=model_selection.train_test_split(X,Y,test_size=0.2,shuffle=False)
clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
# clf = LinearRegression()
clf.fit(np.array(train_x), np.array(train_y))
predict_y = clf.predict(test_x)
print('accuracy：',clf.score(np.array(test_x), np.array(test_y))) 
print('recall：',recall_score(test_y,clf.predict(test_x),average = 'macro'))
print('accuracy：',precision_score(test_y, clf.predict(test_x), average='macro'))

print('')

print('X_full_tfidf')
X = X_full_tfidf
train_x,test_x,train_y,test_y=model_selection.train_test_split(X,Y,test_size=0.2,shuffle=False)
clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
# clf = LinearRegression()
clf.fit(np.array(train_x), np.array(train_y))
predict_y = clf.predict(test_x)
print('accuracy：',clf.score(np.array(test_x), np.array(test_y))) 
print('recall：',recall_score(test_y,clf.predict(test_x),average = 'macro'))
print('accuracy：',precision_score(test_y, clf.predict(test_x), average='macro'))

# neg pos 词
pos_words = {}
neg_words = {}
for word in sent_words:
   if count[word]['sent'] > 0:
      pos_words[word.lower()] = count[word]['pos']+count[word]['neg']
   else:
      neg_words[word.lower()] = count[word]['pos']+count[word]['neg']

output_cloud(pos_words,'pos')
output_cloud(neg_words,'neg')

# neg pos 词
pos_words = {}
neg_words = {}
for word in count.keys():
   if count[word]['sent'] > 0:
      pos_words[word.lower()] = count[word]['pos']+count[word]['neg']
   else:
      neg_words[word.lower()] = count[word]['pos']+count[word]['neg']

output_cloud(pos_words,'pos')
output_cloud(neg_words,'neg')

## 求于bl词典的覆盖率
bl_sent = {}
bl_pos = my_read('/home/stocksentiment/Dynamic-Financial-News-Collection-and-Analysis/sentiment_analysis/bl/positive.txt')  # 4783
bl_neg = my_read('/home/stocksentiment/Dynamic-Financial-News-Collection-and-Analysis/sentiment_analysis/bl/negative.txt')  # 2006


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
## context sentiment dict
# sent_words = [word.lower() for word in sent_words]
# feature_words = {}
# sentiment_feature = {}

# for data in tqdm(datas):
#     for tags in data['tags']:
#         for word,tag in tags:
#             if tag not in nn or len(word)<3: # vb+nn
#                 continue
#             # word = stem_and_check(word)
#             if word not in feature_words.keys():
#                 feature_words[word] = 1
#             else:
#                 feature_words[word] += 1

# # avg_f = sum([item[1] for item in feature_words.items()])/len(feature_words.keys())
# res = sorted(feature_words.items(),key=lambda feature_words:feature_words[1],reverse=True)
# words = res[:400]
# # for word,value in tqdm(copy.items()):
# #     if value<avg_f+200:
# #         del feature_words[word]

# feature_words = [inf.singularize(word).lower() for word,freq in words]

# sf_len = 0
# for data in tqdm(datas):
#     rate = data['rate']
#     for tokens in data['tokens']:
#         token_dict = {}
#         for token in tokens:
#             token_dict[inf.singularize(token).lower()] = token
#         _tokens = [inf.singularize(token).lower() for token in tokens]

#         for w in list(set(sent_words).intersection(set(_tokens))):
#             for f in list(set(feature_words).intersection(set(_tokens))):
#                 if f != w:
#                     if abs(_tokens.index(w)-_tokens.index(f))<3 and ',' not in data['content'][min(data['content'].index(token_dict[f]),data['content'].index(token_dict[w])):max(data['content'].index(token_dict[f]),data['content'].index(token_dict[w]))]:
#                         sf_len += 1
#                         if f not in sentiment_feature.keys():
#                             sentiment_feature[f] = {}
#                             if rate > 0:
#                                 sentiment_feature[f][w] = {'pos':1,'neg':0}
#                             if rate < 0:
#                                 sentiment_feature[f][w] = {'pos':0,'neg':1}
#                         else:
#                             if w not in sentiment_feature[f].keys():
#                                 sentiment_feature[f][w] = {'pos':0,'neg':0}
#                             if rate > 0:
#                                 sentiment_feature[f][w]['pos'] += 1
#                             if rate < 0:
#                                 sentiment_feature[f][w]['neg'] += 1

# avg_sf = sf_len/len(sentiment_feature.keys())
# copy = sentiment_feature.copy()

# for f,v in tqdm(sentiment_feature.items()):
#     for w,value in v.items():
#         if value['pos']+value['neg']<avg_sf: #avg_sf
#             # print(f,w,value)
#             # del sentiment_feature[f][w]
#             sentiment_feature[f][w]['sent'] = 0
#             continue
#         pos = value['pos']/POS
#         neg = value['neg']/NEG
        
#         value['PD'] = (pos-neg)/(pos+neg) # polarity difference
#         sentiment_feature[f][w]['sent'] = value['PD'] * value['PD'] * np.sign(value['PD'])

# res = sorted(sentiment_feature.items(),key=lambda sentiment_feature:sentiment_feature[1]['sent'],reverse=False)

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
# for sf,value in sentiment_feature.items():
#    if value['pos']+value['neg'] > avg_sf:
#       print(sf,freq)

news2vector(datas,count,bl_sent)

output = open('datas.pkl', 'wb')
pickle.dump(datas, output)
print('dump finish')

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from keras.utils import to_categorical
import matplotlib.pyplot as plt
# RandomForest
X1 = [data['DsVector'] for data in datas]
X2 = [data['SnVector'] for data in datas]
X3 = [data['BlVector'] for data in datas]
X4 = [data['PmiVector'] for data in datas]
X5 = [data['DsVector_rate'] for data in datas]

Xs = {'DsVector':X1,'SnVector':X2,'BlVector':X3,'PmiVector':X4,'DsVector_rate':X5}

Y = [np.sign(data['rate']) for data in datas]
for vectorname in Xs.keys():
    print(vectorname+'==============')
    X = Xs[vectorname]
    train_x,test_x,train_y,test_y = model_selection.train_test_split(X,Y,test_size=0.2,shuffle=False)

    clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)

    clf.fit(np.array(train_x), np.array(train_y))

    predict_y = clf.predict(test_x)
    recall = recall_score(test_y,clf.predict(test_x),average = 'macro')
    precision = precision_score(test_y, clf.predict(test_x), average='macro')

    print('accuracy：',clf.score(np.array(test_x), np.array(test_y))) 
    print('recall：',recall)
    print('precision：',precision)
    print('f1_score',2*recall*precision/(recall+precision))

    vote_predict_y = vote(predict_y,datas[-len(test_x):])
    vote_recall = recall_score(test_y,vote_predict_y,average = 'macro')
    vote_precision = precision_score(test_y, vote_predict_y, average='macro')

    print('voting accuracy：',accuracy(vote_predict_y,test_y))
    print('voting recall：',vote_recall)
    print('voting precision：',vote_precision)
    print('voting f1_score',2*vote_recall*vote_precision/(vote_recall+vote_precision))
    print('')

# GaussianNB
for vectorname in Xs.keys():
    print(vectorname+'==============')
    X = Xs[vectorname]
    train_x,test_x,train_y,test_y = model_selection.train_test_split(X,Y,test_size=0.2,shuffle=False)

    clf = GaussianNB()

    clf.fit(np.array(train_x), np.array(train_y))

    predict_y = clf.predict(test_x)
    recall = recall_score(test_y,clf.predict(test_x),average = 'macro')
    precision = precision_score(test_y, clf.predict(test_x), average='macro')

    print('accuracy：',clf.score(np.array(test_x), np.array(test_y))) 
    print('recall：',recall)
    print('precision：',precision)
    print('f1_score',2*recall*precision/(recall+precision))

    vote_predict_y = vote(predict_y,datas[-len(test_x):])
    vote_recall = recall_score(test_y,vote_predict_y,average = 'macro')
    vote_precision = precision_score(test_y, vote_predict_y, average='macro')

    print('voting accuracy：',accuracy(vote_predict_y,test_y))
    print('voting recall：',vote_recall)
    print('voting precision：',vote_precision)
    print('voting f1_score',2*vote_recall*vote_precision/(vote_recall+vote_precision))
    print('')

# dnn
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical

Y = [np.sign(data['rate']) for data in datas]
for vectorname in Xs.keys():
    print(vectorname+'==============')
    X = Xs[vectorname]
    train_x,test_x,Y1,Y2 = model_selection.train_test_split(X,Y,test_size=0.2,shuffle=False)
    # train_x = [data['DsVector_rate'] for data in datas]
    # Y = [np.sign(data['rate']) for data in datas]
    # test_x = [data['DsVector_rate'] for data in datas2[2000:]]
    # Y2 = [np.sign(data['rate']) for data in datas2[2000:]]

    train_y = []
    for y in Y1:
        if y == 1:
            train_y.append(np.array([0,1]))
        else:    
            train_y.append(np.array([1,0]))

    test_y = []
    for y in Y2:
        if y == 1:
            test_y.append(np.array([0,1]))
        else:    
            test_y.append(np.array([1,0]))

    num_classes = 2
    
#     train_y = to_categorical(train_y,num_classes=num_classes)
#     test_y = to_categorical(test_y,num_classes=num_classes)
    nmodel = Sequential()
    nmodel.add(Dense(units=num_classes, activation = 'relu', input_dim = np.array(train_x).shape[1]))
    nmodel.add(Dropout(0.5))
    nmodel.add(Dense(2, activation = 'relu'))
    nmodel.add(Dropout(0.5))
    # dropout:https://blog.csdn.net/program_developer/article/details/80737724
    nmodel.add(Dense(2, activation = 'softmax'))
    nmodel.compile(loss = 'categorical_crossentropy',
                   optimizer = 'adam',
                   metrics = ['accuracy'])
    nmodel.fit(np.array(train_x),np.array(train_y),epochs=10, batch_size=5)
    print(nmodel.evaluate(np.array(test_x),np.array(test_y), batch_size=5))

## rnn preprocessing
rnn_dict = {}
date_set = set([data['date'] for data in datas])
company_set = set([data['company'] for data in datas])
for company in company_set:
    rnn_dict[company] = {}
    for date in date_set:
        rnn_dict[company][date] = set()
for idx in range(0,len(datas)):
    date = datas[idx]['date']
    company = datas[idx]['company']         
    rnn_dict[company][date].add(idx)

res = sorted(apple_dict.items(),key=lambda apple_dict:apple_dict[0],reverse=True)

for idx in tqdm(range(0,len(datas))):
    date = datas[idx]['date']
    company = datas[idx]['company']
    datas[idx]['rnn_vector'] = []
    for i in range(0,3):
        average_vector = []
        try:
            day_vector = rnn_dict[company][time_fun(date,i)]
        except:
            datas[idx]['rnn_vector'].append([0,0,0,0])
            continue
        vector0 = [datas[data_idx]['DsVector_rate'][0] for data_idx in day_vector]
        vector1 = [datas[data_idx]['DsVector_rate'][1] for data_idx in day_vector]
        vector2 = [datas[data_idx]['DsVector_rate'][2] for data_idx in day_vector]
        vector3 = [datas[data_idx]['DsVector_rate'][3] for data_idx in day_vector]
        try:
            average_vector.append(sum(vector0)/len(day_vector)) 
        except:
            average_vector.append(0)
        try:
            average_vector.append(sum(vector1)/len(day_vector)) 
        except:
            average_vector.append(0)
        try:
            average_vector.append(sum(vector2)/len(day_vector)) 
        except:
            average_vector.append(0)
        try:
            average_vector.append(sum(vector3)/len(day_vector)) 
        except:
            average_vector.append(0)
        datas[idx]['rnn_vector'].append(average_vector)
            
# rnn
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical
from keras.layers import LSTM


X = [data['rnn_vector'] for data in datas]
Y = [np.sign(data['rate']) for data in datas]
    
X = np.array(X)
# X = np.concatenate(X, axis=0)

train_x,test_x,Y1,Y2 = model_selection.train_test_split(X,Y,test_size=0.2,shuffle=False)

train_y = []
for y in Y1:
    if y == 1:
        train_y.append([0,1])
    else:    
        train_y.append([1,0])

test_y = []
for y in Y2:
    if y == 1:
        test_y.append([0,1])
    else:    
        test_y.append([1,0])

num_classes = 2
neurons = 2                 
activation_function = 'tanh'  
loss = 'mse'                  
optimizer="adam"              
dropout = 0.25                 
batch_size = 12               
epochs = 53

model = Sequential()
model.add(LSTM(2, input_shape=(train_x.shape[1], train_x.shape[2])))
model.add(Dropout(dropout))

model.add(Activation(activation_function))
model.compile(loss=loss, optimizer=optimizer, metrics=['mae'])

model.fit(np.array(train_x),np.array(train_y),epochs=5, batch_size=5)

# # model.evaluate(np.array(test_x),np.array(test_y), batch_size=5)
predict_y = model.predict(np.array(test_x))
predict_y = np.argmax(predict_y,axis=1)
# model.predict(test)预测的是数值,而且输出的还是5个编码值，不过是实数，预测后要经过argmax(predict_test,axis=1)

# company Apple
appledatas = [data for data in datas if data['company'] =='Apple Inc.']
apple_count = train_sent_dict(appledatas)
news2vector(appledatas,apple_count,bl_sent)

apple_X1 = [data['DsVector'] for data in appledatas]
apple_X2 = [data['SnVector'] for data in appledatas]
apple_X3 = [data['BlVector'] for data in appledatas]
apple_X4 = [data['PmiVector'] for data in appledatas]
# X = [data['ContextVector'] for data in datas]
apple_X5 = [data['DsVector_rate'] for data in appledatas]

apple_Xs = {'DsVector':apple_X1,'SnVector':apple_X2,'BlVector':apple_X3,'PmiVector':apple_X4,'DsVector_rate':apple_X5}

Y = [np.sign(data['rate']) for data in appledatas]

for vectorname in apple_Xs.keys():
    print(vectorname+'==============')
    X = apple_Xs[vectorname]
    train_x,test_x,train_y,test_y = model_selection.train_test_split(X,Y,test_size=0.2,shuffle=False)

    clf = GaussianNB()
    
    clf.fit(np.array(train_x), np.array(train_y))
    predict_y = clf.predict(test_x)
    recall = recall_score(test_y,clf.predict(test_x),average = 'macro')
    precision = precision_score(test_y, clf.predict(test_x), average='macro')

    print('recall：',recall)
    print('precision：',precision)
    print('f1_score',2*recall*precision/(recall+precision))

    vote_predict_y = vote(predict_y,datas[-len(test_x):])
    vote_recall = recall_score(test_y,vote_predict_y,average = 'macro')
    vote_precision = precision_score(test_y, vote_predict_y, average='macro')

    print('voting accuracy：',accuracy(vote_predict_y,test_y))
    print('voting recall：',vote_recall)
    print('voting precision：',vote_precision)
    print('voting f1_score',2*vote_recall*vote_precision/(vote_recall+vote_precision))
    print('')

# company Apple Randomforest

for vectorname in Xs.keys():
    print(vectorname+'==============')
    X = apple_Xs[vectorname]
    train_x,test_x,train_y,test_y = model_selection.train_test_split(X,Y,test_size=0.2,shuffle=False)

    clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
    
    clf.fit(np.array(train_x), np.array(train_y))
    predict_y = clf.predict(test_x)
    recall = recall_score(test_y,clf.predict(test_x),average = 'macro')
    precision = precision_score(test_y, clf.predict(test_x), average='macro')

    print('recall：',recall)
    print('precision：',precision)
    print('f1_score',2*recall*precision/(recall+precision))

    vote_predict_y = vote(predict_y,datas[-len(test_x):])
    vote_recall = recall_score(test_y,vote_predict_y,average = 'macro')
    vote_precision = precision_score(test_y, vote_predict_y, average='macro')

    print('voting accuracy：',accuracy(vote_predict_y,test_y))
    print('voting recall：',vote_recall)
    print('voting precision：',vote_precision)
    print('voting f1_score',2*vote_recall*vote_precision/(vote_recall+vote_precision))
    print('')

# company Apple DNN
for vectorname in Xs.keys():
    print(vectorname+'==============')
    X = apple_Xs[vectorname]
    train_x,test_x,Y1,Y2 = model_selection.train_test_split(X,Y,test_size=0.2,shuffle=False)
    # train_x = [data['DsVector_rate'] for data in datas]
    # Y = [np.sign(data['rate']) for data in datas]
    # test_x = [data['DsVector_rate'] for data in datas2[2000:]]
    # Y2 = [np.sign(data['rate']) for data in datas2[2000:]]

    train_y = []
    for y in Y1:
        if y == 1:
            train_y.append(np.array([0,1]))
        else:    
            train_y.append(np.array([1,0]))

    test_y = []
    for y in Y2:
        if y == 1:
            test_y.append(np.array([0,1]))
        else:    
            test_y.append(np.array([1,0]))

    num_classes = 2
    
#     train_y = to_categorical(train_y,num_classes=num_classes)
#     test_y = to_categorical(test_y,num_classes=num_classes)
    nmodel = Sequential()
    nmodel.add(Dense(units=num_classes, activation = 'relu', input_dim = np.array(train_x).shape[1]))
    nmodel.add(Dropout(0.5))
    nmodel.add(Dense(2, activation = 'relu'))
    nmodel.add(Dropout(0.5))
    # dropout:https://blog.csdn.net/program_developer/article/details/80737724
    nmodel.add(Dense(2, activation = 'softmax'))
    nmodel.compile(loss = 'categorical_crossentropy',
                   optimizer = 'adam',
                   metrics = ['accuracy'])
    nmodel.fit(np.array(train_x),np.array(train_y),epochs=10, batch_size=5)
    print(nmodel.evaluate(np.array(test_x),np.array(test_y), batch_size=5))

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical
X = [for data in datas]
Y = [np.sign(data['rate']) for data in datas]
for vectorname in apple_Xs.keys():
    print(vectorname+'==============')
    X = apple_Xs[vectorname]
    train_x,test_x,Y1,Y2 = model_selection.train_test_split(X,Y,test_size=0.2,shuffle=False)
    # train_x = [data['DsVector_rate'] for data in datas]
    # Y = [np.sign(data['rate']) for data in datas]
    # test_x = [data['DsVector_rate'] for data in datas2[2000:]]
    # Y2 = [np.sign(data['rate']) for data in datas2[2000:]]

    train_y = []
    for y in Y1:
        if y == 1:
            train_y.append(np.array([0,1]))
        else:    
            train_y.append(np.array([1,0]))

    test_y = []
    for y in Y2:
        if y == 1:
            test_y.append(np.array([0,1]))
        else:    
            test_y.append(np.array([1,0]))

    num_classes = 2
    model = Sequential()
    model.add(LSTM(neurons, return_sequences=True, 
    input_shape=(inputs.shape[1], inputs.shape[2]), activation=activ_func))
    model.add(Dropout(dropout))
    model.add(LSTM(neurons, return_sequences=True, activation=activ_func))
    model.add(Dropout(dropout))
    model.add(LSTM(neurons, activation=activ_func))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))
    model.compile(loss=loss, optimizer=optimizer, metrics=['mae'])
    model.summary()