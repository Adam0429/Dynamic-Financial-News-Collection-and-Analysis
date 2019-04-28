import nltk
import re
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
import json
import IPython
# nltk.download()

def review_to_words(review_text):
	# 1. Remove HTML
	# review_text = BeautifulSoup(raw_review).get_text() 
	#
	# 2. Remove non-letters		
	letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
	#
	# 3. Convert to lower case, split into individual words
	words = letters_only.lower().split()							 
	#
	# 4. In Python, searching a set is much faster than searching
	#   a list, so convert the stop words to a set
	_stopwords = set(stopwords.words("english"))
	_stopwords = nltk.corpus.stopwords.words('english')
	_stopwords.append('would')
	_stopwords.append('kmh')
	_stopwords.append('mph')
	_stopwords.append('  ')
	_stopwords.append('Reuters')				  
	# 
	# 5. Remove stop words
	meaningful_words = [w for w in words if not w in _stopwords]   
	#
	# 6. Join the words back into one string separated by space, 
	# and return the result.
	return(meaningful_words)

def PS(w,contents,labels):
	N = len(contents)
	pos_count = 0
	neg_count = 0
	w_pos_count = 0
	w_neg_count = 0
	w_count = 0
	for index in range(0,len(labels)):
		if w in contents[index]:
			w_count += 1
		label = labels[index].split(',')
		if '暂无数据' in label:
			continue
		elif '1' in label and '0' not in label:
			pos_count += 1
			if w in contents[index]:
				w_pos_count += 1
			continue
		elif '0' in label and '1' not in label:
			neg_count += 1
			if w in contents[index]:
				w_neg_count += 1

	freq_pos = pos_count/N 
	freq_neg = neg_count/N
	freq_w_pos = w_pos_count/N
	freq_w_neg = w_neg_count/N
	freq_w = w_count/N

	if freq_w_pos*N == 0:
		PMI_w_pos = 0
	else:
		PMI_w_pos = np.log(freq_w_pos*N/freq_w*freq_pos)
	if freq_w_neg*N == 0:
		PMI_w_neg = 0
	else:
		PMI_w_neg = np.log(freq_w_neg*N/freq_w*freq_neg)
	return PMI_w_pos - PMI_w_neg



workbook = xlrd.open_workbook(r'/Users/wangfeihong/Desktop/Dynamic-Financial-News-Collection-and-Analysis/data/labeled_data2018.xls')
sheet = workbook.sheet_by_index(0)
contents = sheet.col_values(1)
prices = sheet.col_values(3)

len_word = 0
tokens = []
sentences = []
labels = []
for price in prices:
	# if '暂无数据' in label:
	# 	labels.append(0)
	price_list = json.loads(price)
	up = price_list[2] - price_list[1]
	if up > 0:
		labels.append(1)		
	else:
		labels.append(0)
for content in tqdm(contents):
	# doc = nlp(content)
	# sents = list(doc.sents)   # 分解为句子
	# for sent in sents:
	# 	sentences.append(str(sents))
	# 	token = list(map(str,sent))
	# 	tokens.append(token)
	sentences.append(content)
	token = review_to_words(content)
	tokens.append(token)

model = Word2Vec(sentences = tokens,min_count = 2)

bag_of_keywords = set(['rise','drop','fall','gain','surge','shrink','jump','slump','surge'])
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
bok_count = CountVectorizer(lowercase=False,min_df=1, vocabulary=bag_of_keywords)
X_bok_count = bok_count.fit_transform(sentences)
X_bok_count = X_bok_count.toarray()

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

ct_count = CountVectorizer(lowercase = False, min_df = 1, vocabulary = category_tags)
X_ct_count = ct_count.fit_transform(sentences)
X_ct_count = X_ct_count.toarray()

ct_tfidf = TfidfVectorizer(lowercase = False, min_df = 1, vocabulary = category_tags)
X_ct_idf = ct_tfidf.fit_transform(sentences)
X_ct_idf = X_ct_idf.toarray()

full_tfidf = TfidfVectorizer(lowercase=False, min_df = 1,vocabulary=bag_of_keywords,use_idf=False)
X_full_tfidf = full_tfidf.fit_transform(sentences)
X_full_tfidf = X_full_tfidf.toarray()


num_classes = 2


# x = np.random.random((664,200))
# y = np.random.random((664, 10))


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

x = X_bok_count
y = np.array(labels)
y = to_categorical(y,num_classes=2)
train_x,test_x,train_y,test_y=model_selection.train_test_split(x,y,test_size=0.2,shuffle=False)
clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
# clf = LinearRegression()
clf.fit(np.array(train_x), np.array(train_y))
predict_y = clf.predict(test_x)
print('准确率：',clf.score(np.array(test_x), np.array(test_y))) 
print('召回率：',recall_score(test_y,clf.predict(test_x),average = 'macro'))
print('精确率：',precision_score(test_y, clf.predict(test_x), average='macro'))

nmodel = Sequential()
nmodel.add(Dense(units=num_classes, activation = 'relu', input_dim = x.shape[1]))
# 输出就是在输出层有几个神经元,每个神经元代表着一个预测结果,label的序列长度为十，须要十个神经元与之对应。label用to_categorical转换
nmodel.add(Dropout(0.5))
#避免过拟合
''' 多层的意义
单层神经网络只能用于表示线性可分离的函数。也就是说非常简单的问题，例如，分类问题中可以被一行整齐地分隔开的两个类。如果你的问题相对简单，那么单层网络就足够了。
然而，我们有兴趣解决的大多数问题都不是线性可分的。
多层感知器可用于表示凸区域。这意味着，实际上，他们可以学习在一些高维空间中围绕实例绘制形状，以对它们进行分类，从而克服线性可分性的限制。'''
nmodel.add(Dense(2, activation = 'relu'))
nmodel.add(Dropout(0.5))
# dropout:https://blog.csdn.net/program_developer/article/details/80737724
nmodel.add(Dense(2, activation = 'softmax'))
nmodel.compile(loss = 'categorical_crossentropy',
               optimizer = 'adam',
               metrics = ['accuracy'])
#verbose=1:更新日志 verbose=2:每个epoch一个进度行

nmodel.fit(train_x,train_y,epochs=10, batch_size=5)
score = nmodel.evaluate(test_x, test_y, batch_size=20)
print(score)

# predict = nmodel.predict_classes(x_test,batch_size=5)

# for i in range(predict)

import IPython
IPython.embed()

# for row in rows:
# 	rows[rows.index(row)] = review_to_words(row)
# vectorizer = CountVectorizer(analyzer = "word",   
#							  tokenizer = None,	
#							  preprocessor = None, 
#							  stop_words = None,  
#							  max_features = 10) 

# train_data_features = vectorizer.fit_transform(rows)
# train_data_features = train_data_features.toarray()
# vocab = vectorizer.get_feature_names()
# gnb = GaussianNB()
# gnb.fit(train_data_features, [1]*100)
# r = gnb.predict([np.ones(10)])
# print(r)
