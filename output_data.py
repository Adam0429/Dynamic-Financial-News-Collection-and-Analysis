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

path2018 = r'/Users/wangfeihong/Desktop/Dynamic-Financial-News-Collection-and-Analysis/data/labeled_data2018.xls'

workbook = xlrd.open_workbook(path2018)
worksheet = workbook.sheet_by_index(0)
titles = worksheet.col_values(0)
contents = worksheet.col_values(1)
companies = worksheet.col_values(2)
prices = worksheet.col_values(3)
dates = worksheet.col_values(4)

rates = []
score_list = []


datas = []
for i in tqdm(range(0,len(contents))):
    # if '*' not in contents[i]:
        # if companies[i] != 'Apple Inc.':
        #     continue
    price_list = json.loads(prices[i])
    data = {}
    data['title'] = titles[i]
    data['content'] = contents[i]
    data['company'] = companies[i]
    data['rate'] = (price_list[2]-price_list[1])/price_list[1]
    data['date'] = dates[i]
    data['sent'] = sentiment_score(data['content'])
    datas.append(data)


with open("data.json","w") as f:
    json.dump(datas,f)
IPython.embed()
