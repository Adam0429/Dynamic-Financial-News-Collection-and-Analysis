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
	stops = set(stopwords.words("english"))				  
	# 
	# 5. Remove stop words
	meaningful_words = [w for w in words if not w in stops]   
	#
	# 6. Join the words back into one string separated by space, 
	# and return the result.
	return(' '.join(meaningful_words))



workbook = xlrd.open_workbook(r'/Users/wangfeihong/Desktop/Dynamic-Financial-News-Collection-and-Analysis/data/data2.xls')
sheet = workbook.sheet_by_index(0)
contents = sheet.col_values(4)[1:]
nlp = spacy.load('en')
tokens = []
sentences = []
for content in tqdm(contents):
	doc = nlp(content)
	sents = list(doc.sents)   # 分解为句子
	for sent in sents:
		sentences.append(str(sents))
		token=list(map(str,sent))
		tokens.append(token)

model = Word2Vec(sentences = tokens,min_count = 2)
bag_of_keywords = set(['rise','drop','fall','gain','surge','shrink','jump','slump','surge'])
stop = False
bok_size = 1000
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

bag_of_keywords = np.array(list(bag_of_keywords))
bok_tfidf = TfidfVectorizer(lowercase = False, min_df = 1, vocabulary=bag_of_keywords)
X_bok_tfidf = bok_tfidf.fit_transform(sentences)
X_bok_tfidf = X_bok_tfidf.toarray()
import IPython
IPython.embed()
bok_count = CountVectorizer(lowercase=False,min_df=1,vocabulary=bag_of_keywords)
X_bok_count = bok_count.fit_transform(sentences)
X_bok_count = X_bok_count.toarray()
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