import quandl

quandl.ApiConfig.api_key = 'J2KokzoYT_8FqwQzbx9_'
df = quandl.get("WIKI/GOOGL",rows=5)
# df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
print(df)
# print(df)