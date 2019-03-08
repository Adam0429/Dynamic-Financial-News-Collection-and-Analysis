import xlrd,xlwt
import glob
import IPython
path = r'/Users/wangfeihong/Desktop/Dynamic-Financial-News-Collection-and-Analysis/data/*label.xls'
titles = []
news = []
companys = []
prices = []
for file in glob.glob(path):
	workbook = xlrd.open_workbook(file)
	sheet = workbook.sheet_by_index(0)
	titles += sheet.col_values(0)
	news += sheet.col_values(1)
	companys += sheet.col_values(2)
	prices += sheet.col_values(3)
