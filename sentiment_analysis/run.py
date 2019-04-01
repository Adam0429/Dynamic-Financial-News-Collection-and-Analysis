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
# score_list = []
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

# datas = datas[:17687]
# datas = datas[-2000:]

count = {}

POS = 0
NEG = 0

pos_count = 0
neg_count = 0

N = 0 #len of tokens


for data in tqdm(datas): # datas[:17687]
    tokens = data['tokens']
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
            else:
                count[token] = {'pos':1,'neg':0} 

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
            else:
                count[token] = {'pos':0,'neg':1} 




## freq
copy = count.copy()
sent_words = [] # PD>0.3情感值

freq_pos = pos_count/len(datas) 
freq_neg = neg_count/len(datas)

# DS sent and PMI sent
for word,value in tqdm(copy.items()):
    if value['pos']+value['neg']<10:
        del count[word]
        continue
    pos = value['pos']/len(datas)
    neg = value['neg']/len(datas)


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
    token_dict = {}
    for token in data['tokens']:
        token_dict[inf.singularize(token).lower()] = token

    
    for w in sent_words:
        if w not in tokens:
            continue
        for f in feature_words:
            if f in tokens and f != w:
                if abs(tokens.index(w)-tokens.index(f))<3 and ',' not in data['content'][min(data['content'].index(token_dict[f]),data['content'].index(token_dict[w])):max(data['content'].index(token_dict[f]),data['content'].index(token_dict[w]))]:
                    sf_len += 1
                    if f not in sentiment_feature.keys():
                        sentiment_feature[f] = {}
                        if rate > 0:
                            sentiment_feature[f][w] = {'pos':1,'neg':0}
                        if rate < 0:
                            sentiment_feature[f][w] = {'pos':0,'neg':1}
                    else:
                        if w not in sentiment_feature[f].keys():
                            sentiment_feature[f][w] = {'pos':0,'neg':0}
                        if rate > 0:
                            sentiment_feature[f][w]['pos'] += 1
                        if rate < 0:
                            sentiment_feature[f][w]['neg'] += 1

# avg_sf = sf_len/len(sentiment_feature.keys())
# copy = sentiment_feature.copy()

for f,v in tqdm(sentiment_feature.items()):
    for w,value in v.items():
        if value['pos']+value['neg']<1: #avg_sf
            # print(f,w,value)
            # del sentiment_feature[f][w]
            sentiment_feature[f][w]['sent'] = 0
            continue
        pos = value['pos']/POS
        neg = value['neg']/NEG
        
        value['PD'] = (pos-neg)/(pos+neg) # polarity difference
        sentiment_feature[f][w]['sent'] = value['PD'] * value['PD'] * np.sign(value['PD'])

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
# # for sf,value in sentiment_feature.items():
#    if value['pos']+value['neg'] > avg_sf:
#       print(sf,freq)

# Predict
# content to vector

for data in tqdm(datas):
    idx = datas.index(data)
    tags = data['tags']
    tokens = data['tokens']
    datas[idx]['DsVector'] = [0,0,0,0]
    datas[idx]['SnVector'] = [0,0,0,0]
    datas[idx]['BlVector'] = [0,0,0,0]
    datas[idx]['PmiVector'] = [0,0,0,0]
    datas[idx]['ContextVector'] = [0,0,0,0]


    for f in [token for token in tokens if token in sentiment_feature.keys()]:
        for w in sentiment_feature[f].keys():
            if w in tokens:
                if abs(tokens.index(f)-tokens.index(w))<3 and ',' not in data['content'][min(data['content'].index(f),data['content'].index(w)):max(data['content'].index(f),data['content'].index(w))]:
                    if tags[tokens.index(f)][1] in adj:
                        if f in count.keys():
                            datas[idx]['ContextVector'][0] += count[f]['sent']
                    elif tags[tokens.index(f)][1] in adv:
                        if f in count.keys():
                            datas[idx]['ContextVector'][1] += count[f]['sent']
                    if tags[tokens.index(w)][1] in nn:
                        if w in count.keys():
                            datas[idx]['ContextVector'][2] = count[w]['sent']
                    elif tags[tokens.index(w)][1] in vb:
                        if w in count.keys():
                            datas[idx]['ContextVector'][3] = count[w]['sent']
                    # print(sentiment_feature[f][w]['sent'])

    for word,tag in tags:
        if tag in adj:
            if word in count.keys():
                datas[idx]['DsVector'][0] += count[word]['sent']
                datas[idx]['PmiVector'][0] += count[word]['PMI_sent']
            if word in sn.data.keys():
                datas[idx]['SnVector'][0] += float(sn.polarity_intense(word))
            if word in bl_sent.keys():
                datas[idx]['BlVector'][0] += bl_sent[word]
        elif tag in adv:
            if word in count.keys():
                datas[idx]['SnVector'][1] += count[word]['sent']
                datas[idx]['PmiVector'][1] += count[word]['PMI_sent']
            if word in sn.data.keys():
                datas[idx]['SnVector'][1] += float(sn.polarity_intense(word))
            if word in bl_sent.keys():
                datas[idx]['BlVector'][0] += bl_sent[word]  
        elif tag in nn:
            if word in count.keys():
                datas[idx]['DsVector'][2] = count[word]['sent']
                datas[idx]['PmiVector'][2] += count[word]['PMI_sent']
            if word in sn.data.keys():
                datas[idx]['SnVector'][2] += float(sn.polarity_intense(word))
            if word in bl_sent.keys():
                datas[idx]['BlVector'][0] += bl_sent[word]
        elif tag in vb:
            if word in count.keys():
                datas[idx]['DsVector'][3] = count[word]['sent']
                datas[idx]['PmiVector'][3] += count[word]['PMI_sent']
            if word in sn.data.keys():
                datas[idx]['SnVector'][3] += float(sn.polarity_intense(word))
            if word in bl_sent.keys():
                datas[idx]['BlVector'][0] += bl_sent[word]
    # datas[idx]['DsVector'] = [adv_score,adv_score,noun_score,verb_score]

from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from keras.utils import to_categorical

X = [data['DsVector'] for data in datas]
# X = [data['SnVector'] for data in datas]
# X = [data['BlVector'] for data in datas]
# X = [data['PmiVector'] for data in datas]
# X = [data['ContextVector'] for data in datas]
Y = [np.sign(data['rate']) for data in datas]

# print(len(X)-X.count([0,0,0,0])) 403
# x_train,x_test,y_train,y_test = model_selection.train_test_split(X,Y,test_size=0.1,shuffle=True)
x_train = X[:17687]
y_train = Y[:17687]
x_test = X[-1000:]
y_test = Y[-1000:] #不用测试数据训练情感词典
clf = GaussianNB()
clf.fit(np.array(x_train), np.array(y_train))
print('准确率：',clf.score(np.array(x_test), np.array(y_test))) 
print('召回率：',recall_score(y_test,clf.predict(x_test),average = 'macro'))
print('精确率：',precision_score(y_test, clf.predict(x_test), average='macro'))

IPython.embed()

accuracy_scores = 0
recall_scores = 0
precision_scores = 0
f1_scores = 0
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
    predict_y = clf.predict(test_x)
    accuracy = clf.score(test_x, test_y)
    accuracy_scores += accuracy
    recall = recall_score(test_y,predict_y,average = 'macro')
    recall_scores += recall
    precision = precision_score(test_y, predict_y, average='macro')
    precision_scores += precision
    f1 = f1_score(test_y,predict_y,average = 'macro')
    f1_scores += f1

print('平均准确率：',accuracy_scores/10)
print('平均召回率：',recall_scores/10)
print('平均精确率：',precision_scores/10)
print('平均F-measure：',f1_scores/10)




## dnn
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Dropout
# from keras.utils import to_categorical

# x_train,x_test,y_train,y_test = model_selection.train_test_split(X,Y,test_size=0.1,shuffle=True)

# num_classes = 2
# x_train = np.array(x_train)
# y_train = np.array(y_train)
# x_test = np.array(x_test)
# y_test = np.array(y_test)

# y_train = to_categorical(y_train,num_classes=2)
# y_test = to_categorical(y_test,num_classes=2)

# nmodel = Sequential()
# nmodel.add(Dense(units=num_classes, activation = 'relu', input_dim = x_train.shape[1]))
# nmodel.add(Dropout(0.5))
# nmodel.add(Dense(2, activation = 'relu'))
# nmodel.add(Dropout(0.5))
# # dropout:https://blog.csdn.net/program_developer/article/details/80737724
# nmodel.add(Dense(2, activation = 'softmax'))
# nmodel.compile(loss = 'categorical_crossentropy',
#                optimizer = 'adam',
#                metrics = ['accuracy'])
# nmodel.fit(x_train,y_train,epochs=10, batch_size=5)
# nmodel.evaluate(x_test,y_test, batch_size=5)


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