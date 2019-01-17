import datetime
import pandas as pd
import pandas_datareader.data as web

def rise_or_fall(stockname):
	today = datetime.date.today()
	yesterday = today - datetime.timedelta(days=1)
	prices = web.DataReader('LNT', 'yahoo', yesterday,today)
	if prices['Close'][1]-prices['Close'][0] > 0:
		return 1
	else:
		return 0
