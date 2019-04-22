from stock_relation import _list
import xlrd,xlwt
import datetime
import pandas as pd
import pandas_datareader.data as web
from tqdm import tqdm
import IPython
import glob
# import spacy

def recent_price(stockname,start_time):
	# today = datetime.date.today()
	date = start_time.split('-')
	t1 = pd.Timestamp(datetime.datetime(int(date[0]),int(date[1]),int(date[2]),0,0,0))  # 创建一个datetime.datetime
	# High            155.789993
	# Low             154.600006
	# Open            155.570007
	# Close           155.720001
	# Volume       521800.000000
	# Adj Close       153.603760
	try:
		prices = web.DataReader(stockname, 'yahoo')
		today_index = list(prices.index).index(t1)
		tomorrow_index = prices.index[today_index+1]
		recent_indexs = [prices.index[today_index+i] for i in range(-1,6)]
		# 前一天 当天 和 后五天的 
		data = []
		for index in recent_indexs:
			data.append(prices.loc[index]['Close'])
	except KeyboardInterrupt:
		exit()
	except:
		return 0
	return data

# path = r'/Users/wangfeihong/Desktop/Dynamic-Financial-News-Collection-and-Analysis/data/data2.xls'
path = r'data*.xls'

_dateMillis = set()

workbook2 = xlwt.Workbook(encoding = 'utf-8')
worksheet2 = workbook2.add_sheet('label',cell_overwrite_ok=True)
count = 0

for path in tqdm(glob.glob(path)):
	try:
		workbook = xlrd.open_workbook(path)
		worksheet = workbook.sheet_by_index(0)
		
		titles = worksheet.col_values(0)[1:]
		dateMillis = worksheet.col_values(3)[1:]
		contents = worksheet.col_values(4)[1:]
		dates = worksheet.col_values(5)[1:]
		for idx in tqdm(range(0,len(contents))):
			# worksheet2.write(idx,1,contents[idx])
			if dateMillis[idx] in _dateMillis:
					continue
			else:
				_dateMillis.add(dateMillis[idx])
			# 去重
			# sents = sent_tokenize(contents[idx])
			# for sent in sents:			
			for relations in _list:
				for item in relations[1:len(relations)-1]: # 不按“领域”标注公司
					if item in contents[idx]:
						recent_prices = recent_price(relations[0],dates[idx])
						# if recent_prices != None:
						worksheet2.write(count,0,titles[idx])
						worksheet2.write(count,1,contents[idx])
						worksheet2.write(count,2,relations[1])
						worksheet2.write(count,3,str(recent_prices))
						worksheet2.write(count,4,dates[idx])
						count += 1
						break
	except:
		print(path+' pass')


workbook2.save('labeled_data.xls')





