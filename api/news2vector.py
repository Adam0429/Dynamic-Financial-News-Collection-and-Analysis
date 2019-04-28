import pickle
import nltk
from tqdm import tqdm
import random
import numpy as np
import re
from nltk.corpus import stopwords
import json
from gensim.models import Word2Vec
from nltk import sent_tokenize

adj = ['JJ','JJR','JJS']
adv = ['RB','RBR','RBS']
vb = ['VB','VBD','VBG','VBN','VBP','VBZ']
nn = ['NN','NNS']

model = open('api/sent_dict.pkl', 'rb')
count = pickle.load(model)

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

def news2vector(news):
	vector = [0,0,0,0]
	sents = sent_tokenize(news)
	tokens = []
	tags = []
	for sent in sents:
		token = review_to_words(sent) # 去停用词会影响词性标注吗？？
		tokens.append(token)
		tags.append(nltk.pos_tag(token))
	for tgs in tags:
		for word,tag in tgs:
			if tag in adj:
				if word in count.keys():
					vector[0] += count[word]['sent_rate']
			elif tag in adv:
				if word in count.keys():
					vector[1] += count[word]['sent_rate']
			elif tag in nn:
				if word in count.keys():
					vector[2] = count[word]['sent_rate']
			elif tag in vb:
				if word in count.keys():
					vector[3] = count[word]['sent_rate']
	return vector


