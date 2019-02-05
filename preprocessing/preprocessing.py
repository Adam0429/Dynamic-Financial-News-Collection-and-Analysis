import nltk
import re
from nltk.tokenize import WordPunctTokenizer  
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
import xlrd,xlwt
import numpy as np
import spacy
from gensim.models import Word2Vec
from tqdm import tqdm
import math
# nltk.download()

def review_to_words(review_text):
	# Function to convert a raw review to a string of words
	# The input is a single string (a raw movie review), and 
	# the output is a single string (a preprocessed movie review)
	#
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
	stopwords = set(stopwords.words("english"))
	stopwords = nltk.corpus.stopwords.words('english')
	stopwords.append('would')
	stopwords.append('kmh')
	stopwords.append('mph')
	stopwords.append('  ')
	stopwords.append('Reuters')				  
	# 
	# 5. Remove stop words
	meaningful_words = [w for w in words if not w in stopwords]   
	#
	# 6. Join the words back into one string separated by space, 
	# and return the result.
	return(' '.join(meaningful_words))

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



workbook = xlrd.open_workbook(r'/Users/wangfeihong/Desktop/Dynamic-Financial-News-Collection-and-Analysis/data/data2_label.xls')
sheet = workbook.sheet_by_index(0)
contents = sheet.col_values(1)[1:]
labels = sheet.col_values(3)[1:]
nlp = spacy.load('en')
len_word = 0
tokens = []
sentences = []
for content in tqdm(contents):
	doc = nlp(content)
	sents = list(doc.sents)   # 分解为句子
	for sent in sents:
		sentences.append(str(sents))
		token = list(map(str,sent))
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

## Bag of keywords
bag_of_keywords = np.array(list(bag_of_keywords))
bok_tfidf = TfidfVectorizer(lowercase = False, min_df = 1, vocabulary=bag_of_keywords)
X_bok_tfidf = bok_tfidf.fit_transform(sentences)
X_bok_tfidf = X_bok_tfidf.toarray()
bok_count = CountVectorizer(lowercase=False,min_df=1,vocabulary=bag_of_keywords)
X_bok_count = bok_count.fit_transform(sentences)
X_bok_count = X_bok_count.toarray()

# print(PS('rise',contents,labels))
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