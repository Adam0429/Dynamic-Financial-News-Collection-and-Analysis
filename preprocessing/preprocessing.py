import nltk
import re
from nltk.tokenize import WordPunctTokenizer  
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
import xlrd,xlwt
import numpy as np
 
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

workbook = xlrd.open_workbook(r'/Users/wangfeihong/Desktop/Financial-Portfolio-Management-using-Reinforcement-Learning/data.xls')
sheet = workbook.sheet_by_index(0)
rows = sheet.col_values(0)[1:]
for row in rows:
	rows[rows.index(row)] = review_to_words(row)
vectorizer = CountVectorizer(analyzer = "word",   
                             tokenizer = None,    
                             preprocessor = None, 
                             stop_words = None,  
                             max_features = 10) 

train_data_features = vectorizer.fit_transform(rows)
train_data_features = train_data_features.toarray()
vocab = vectorizer.get_feature_names()
gnb = GaussianNB()
gnb.fit(train_data_features, [1]*100)
r = gnb.predict([np.ones(10)])
print(r)