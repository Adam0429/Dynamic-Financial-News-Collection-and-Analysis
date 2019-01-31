import nltk
import re
from nltk.tokenize import WordPunctTokenizer  
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
import xlrd,xlwt
import numpy as np
import spacy
from gensim.models import Word2Vec

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
sents = []
for content in contents:
	doc = nlp(content)
	sentences = list(doc.sents)   # 分解为句子
	for sentence in sentences:
		sentence = str(sentence)
		sents.append(sentence)
model = Word2Vec(sentences = sents,min_count = 2)
import IPython
IPython.embed()
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
# for row in rows:
# 	rows[rows.index(row)] = review_to_words(row)
# vectorizer = CountVectorizer(analyzer = "word",   
#                              tokenizer = None,    
#                              preprocessor = None, 
#                              stop_words = None,  
#                              max_features = 10) 

# train_data_features = vectorizer.fit_transform(rows)
# train_data_features = train_data_features.toarray()
# vocab = vectorizer.get_feature_names()
# gnb = GaussianNB()
# gnb.fit(train_data_features, [1]*100)
# r = gnb.predict([np.ones(10)])
# print(r)