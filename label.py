from stock_relation import _list
import xlrd,xlwt
import datetime
import pandas as pd
import pandas_datareader.data as web
from tqdm import tqdm

def rise_or_fall(stockname):
	today = datetime.date.today()
	yesterday = today - datetime.timedelta(days=1)
	prices = web.DataReader(stockname, 'yahoo', yesterday,today)
	if prices['Close'][1]-prices['Close'][0] > 0:
		return '1'
	else:
		return '0'

path = r'data/data.xls'

workbook = xlrd.open_workbook(path)
worksheet = workbook.sheet_by_index(0)

workbook2 = xlwt.Workbook(encoding = 'utf-8')
worksheet2 = workbook2.add_sheet('label',cell_overwrite_ok=True)

titles = worksheet.col_values(0)[1:]
contents = worksheet.col_values(4)[1:]

for idx in tqdm(range(0,len(titles))):
	stocks = ''
	label = []
	worksheet2.write(idx,0,titles[idx])
	worksheet2.write(idx,1,contents[idx])
	for relations in _list:
		for item in relations[1:]:
			if item in contents[idx]:
				stocks += relations[1] + '    '
				label.append(rise_or_fall(relations[0]))
				break
	worksheet2.write(idx,2,stocks)
	worksheet2.write(idx,3,','.join(label))

			
workbook2.save('data_label.xls')


