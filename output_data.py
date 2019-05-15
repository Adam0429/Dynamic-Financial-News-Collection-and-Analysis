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
from gensim.models import Word2Vec
import enchant
import time
from nltk import sent_tokenize

def sentiment_score(text):
	t = TextBlob(text)
	score = t.sentiment.polarity
	return score

path = r'/Users/wangfeihong/Desktop/Dynamic-Financial-News-Collection-and-Analysis/train_data/labeled_data.xls'
# path2019 = r'/Users/wangfeihong/Desktop/Dynamic-Financial-News-Collection-and-Analysis/data/labeled_data2019.xls'

workbook = xlrd.open_workbook(path)
worksheet = workbook.sheet_by_index(0)
titles = worksheet.col_values(0)
contents = worksheet.col_values(1)
companies = worksheet.col_values(2)
prices = worksheet.col_values(3)
dates = worksheet.col_values(4)

rates = []
score_list = []

datas_index = {}
datas = {}

_dates = list(set(dates))
for date in _dates:
	datas[date] = {}


for date in tqdm(_dates):
	for i in range(0,len(dates)):
		if dates[i] == date:
			company = companies[i]
			price_list = json.loads(prices[i])
			if type(price_list) != type([]):
				continue
			rate = (price_list[2]-price_list[1])/price_list[1]
			if company not in datas[date].keys():
				datas[date][company] = {'rate':rate,'sent':sentiment_score(contents[i])}
			else:
				datas[date][company]['sent'] += sentiment_score(contents[i])				

def Normalization(x,Max,Min):
    x = (x - Min) / (Max - Min);
    return x

for date in tqdm(datas.keys()):
	for company in datas[date].keys():
		datas[date][company]['rate_Normalize'] = Normalization(datas[date][company]['rate'],0.1,-0.1)
		datas[date][company]['sent_Normalize'] = Normalization(datas[date][company]['sent'],1,-1)

list_data = []
for date in tqdm(datas.keys()):
	for company in datas[date].keys():
		l = datas[date][company]
		l['company'] = company
		l['date'] = date 
		list_data.append(l)
IPython.embed()
with open("data.json","w") as f:
	 json.dump(list_data,f)
