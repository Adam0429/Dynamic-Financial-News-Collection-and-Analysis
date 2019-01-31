# this script is used to make the stock relations which is used to label news
import xlrd
import re
import functools
import pandas as pd


xls_path = r'/Users/wangfeihong/Desktop/Dynamic-Financial-News-Collection-and-Analysis/data/us_stocks.xls'
dict_path = r'/Users/wangfeihong/Desktop/Dynamic-Financial-News-Collection-and-Analysis/stock_relation.py'
_list = []
badwords = [' Inc.',' Inc',' Corp',' Corp.',' company'] # 这些词会影响匹配
workbook = xlrd.open_workbook(xls_path)
sheet = workbook.sheet_by_index(0)
codes = sheet.col_values(0)[1:]
names = sheet.col_values(1)[1:]
sectors = sheet.col_values(2)[1:]
for idx in range(0,len(codes)):
	name = names[idx]
	items = [codes[idx],name]
	for badword in badwords:
		if badword in name:
			name = name.replace(badword,'')
			items.append(name)
	items.append(sectors[idx])
	_list.append(items)

file = open(dict_path,'w+')
file.write('_list = '+str(_list))
