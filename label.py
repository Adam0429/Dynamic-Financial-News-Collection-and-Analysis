from stock_relation import _list
import xlrd,xlwt
import datetime
import pandas as pd
import pandas_datareader.data as web
from tqdm import tqdm
import spacy

def rise_or_fall(stockname,start_time):
	# today = datetime.date.today()
	st = datetime.datetime.strptime(start_time, '%Y-%m-%d')
	if st.weekday() in [4,5,6]:#周五，六，日
		if st.weekday() == 4:
			et = st - datetime.timedelta(days=-3)
		elif st.weekday() == 5:
			st = st - datetime.timedelta(days=1)
			et = st - datetime.timedelta(days=-3)
		else:
			st = st - datetime.timedelta(days=2)
			et = st - datetime.timedelta(days=-3)

	else:
		st = datetime.datetime.strptime(start_time, '%Y-%m-%d')
		et = st - datetime.timedelta(days=-1)
	end_time = et.strftime("%Y-%m-%d")
	start_time = st.strftime("%Y-%m-%d")
	try:
		prices = web.DataReader(stockname, 'yahoo', start_time,end_time)
		
		if prices['Close'][1]-prices['Close'][0] > 0:
			return '1'
		else:
			return '0'
	except:
		return '暂无数据'

nlp = spacy.load('en')
path = r'/Users/wangfeihong/Desktop/Dynamic-Financial-News-Collection-and-Analysis/data/data3.xls'

workbook = xlrd.open_workbook(path)
worksheet = workbook.sheet_by_index(0)
workbook2 = xlwt.Workbook(encoding = 'utf-8')
worksheet2 = workbook2.add_sheet('label',cell_overwrite_ok=True)

titles = worksheet.col_values(0)[1:]
contents = worksheet.col_values(4)[1:]
dates = worksheet.col_values(5)[1:]
count = 0
for idx in tqdm(range(0,len(contents))):
	stocks = ''
	# worksheet2.write(idx,1,contents[idx])
	doc = nlp(contents[idx])
	sents = list(doc.sents)
	for sent in sents:
		label = []
		sent = str(sent)
		for relations in _list:
			for item in relations[1:]:
				if item in sent:
					stocks += relations[1] + '    '
					label.append(rise_or_fall(relations[0],dates[idx]))
					break
		if len(label) != 0:
			worksheet2.write(count,0,titles[idx])
			worksheet2.write(count,1,sent)
			worksheet2.write(count,2,stocks)
			worksheet2.write(count,3,','.join(label))
			count += 1 

			
workbook2.save('data3_label.xls')


