from stock_relation import _list
import xlrd,xlwt

path = r'/Users/wangfeihong/Desktop/Financial-Portfolio-Management-using-Reinforcement-Learning/data.xls'

workbook = xlrd.open_workbook(path)
worksheet = workbook.sheet_by_index(0)

workbook2 = xlwt.Workbook(encoding = 'utf-8')
worksheet2 = workbook2.add_sheet('label',cell_overwrite_ok=True)

titles = worksheet.col_values(0)[1:]
contents = worksheet.col_values(4)[1:]

for idx in range(0,len(titles)):
	label = ''
	worksheet2.write(idx,0,titles[idx])
	worksheet2.write(idx,1,contents[idx])
	for relations in _list:
		for item in relations[1:]:
			if item in contents[idx]:
				label += relations[1] + '    '
				break
	worksheet2.write(idx,2,label)

			
workbook2.save('data_label.xls')


