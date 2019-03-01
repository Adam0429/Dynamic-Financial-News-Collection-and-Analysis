from senticnet.senticnet import SenticNet
import nltk
from tqdm import tqdm
import xlrd
import pandas as pd
import random
from textblob import TextBlob
import numpy
import IPython

sn = SenticNet()
# concept_info = sn.concept('love')
# polarity_value = sn.polarity_value('love')
# polarity_intense = sn.polarity_intense('love')
# moodtags = sn.moodtags('love')
# semantics = sn.semantics('love')
# sentics = sn.sentics('love') 
 
def sentiment_score_list(text):
    # tokens = nltk.word_tokenize(text)
    # pos_tags = nltk.pos_tag(tokens)
    # score = 0
    # count = 0
    # for word,tag in pos_tags:
    #     if word in sn.data.keys():
    #         score += float(sn.polarity_intense(word))
    #         count += 1
    #         # print(word,sn.polarity_intense(word))
    # if count == 0: #mid
    #     return 2 
    # # if score/count>-0.1 and score/count<0.1:
    # #     return 2
    # elif score/count<0: #neg
    #     return 0
    # elif score/count>0: #pos
    #     return 4
    # if score/count>-0.1 and score/count<0.1:
    #     return 2

    t = TextBlob(text)
    score = t.sentiment.polarity
    if score<0.2: #neg
        return 0
    elif score>0.2: #pos
        return 4


# scores = []
# workbook = pd.read_csv(u'/Users/wangfeihong/Desktop/Dynamic-Financial-News-Collection-and-Analysis/data/sentiment.csv',encoding='ISO-8859-1')
# correct = 0
# pos_count = 0
# neg_count = 0
# pos_correct = 0
# neg_correct = 0

# for i in tqdm(range(0,10000)):
#     i = int(random.random()*1599999)
#     if workbook.loc[i][0] == 0:
#         neg_count += 1
#         if workbook.loc[i][0] == sentiment_score_list(workbook.loc[i][5]):
#             neg_correct += 1
#     elif workbook.loc[i][0] == 4:
#         pos_count += 1
#         if workbook.loc[i][0] == sentiment_score_list(workbook.loc[i][5]):
#             pos_correct += 1
# print('pos',pos_correct/pos_count,pos_count,pos_correct)
# print('neg',neg_correct/neg_count,neg_count,neg_correct)


path = r'/Users/wangfeihong/Desktop/Dynamic-Financial-News-Collection-and-Analysis/data/data_label.xls'

workbook = xlrd.open_workbook(path)
worksheet = workbook.sheet_by_index(0)
contents = worksheet.col_values(1)[1:]
labels = worksheet.col_values(3)[1:]

score_list = []
label_list = []
for i in tqdm(range(0,len(contents))):
    if set(list(labels[i])) == {'1'}:
        score_list.append(sentiment_score_list(contents[i]))
        label_list.append(1)
    if set(list(labels[i])) == {'0'}:
        score_list.append(sentiment_score_list(contents[i]))
        label_list.append(0)

data = {
       'scores':score_list,
       'labels':label_list
       }

df = pd.DataFrame(data)
print(df)
print(df.corr("kendall"))

# pearson：Pearson相关系数来衡量两个数据集合是否在一条线上面，即针对线性数据的相关系数计算，针对非线性                                           数据便会有误差。

# kendall：用于反映分类变量相关性的指标，即针对无序序列的相关系数，非正太分布的数据

# spearman：非线性的，非正太分析的数据的相关系数


# sheet = workbook.sheet_by_index(1)
# labels = sheet.col_values(0)
# contents = sheet.col_values(5)
# for i in tqdm(range(0,contents)):
#     if labels[i] == sentiment_score_list(contents[1]):
#         correct += 1

    # print(content,sentiment_score_list(content))

# newlist =[i for i in scores if i>0.3]
# print(len(newlist))
# print(correct/len(labels))

# score = sentiment_score_list('i love you very much')  
# print(score)