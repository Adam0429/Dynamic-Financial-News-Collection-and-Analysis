import requests
import socks
import socket
import time
import datetime
import time
import IPython
import xlrd,xlwt
import re
from tqdm import tqdm


socks.set_default_proxy(socks.SOCKS5,"127.0.0.1",1080)
socket.socket = socks.socksocket
s = requests.session()
# s.proxies = {"http": "127.0.0.1:1080","https": "127.0.0.1:1080"}
article_url = 'http://www.reuters.com/article/json/data-id'
headers = {
    'accept-encoding': 'gzip, deflate, br',
    'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36',
    'accept': 'application/json, text/javascript, */*; q=0.01',
    'referer': 'https://www.reuters.com/',
    'authority': 'www.reuters.com',
    'x-requested-with': 'XMLHttpRequest',
}
now = int(str(int(time.time()))+'000')
# one_day = 86400
# yesterday = int(time.mktime(datetime.date.today().timetuple())) - one_day # 昨天0点
# yesterday = int(str(yesterday)+'000')
params = (
    ('limit',100),
    ('channel',113)
    # ('endTime',now)
)
# 'http://www.reuters.com/article/json/data-idUSL3N1ZB0RL'
def get(url,headers={},params=()):
    stutas = 1
    while stutas != 200:
        try:
            r = requests.get(url,headers=headers,params=params)
            stutas = r.status_code
        except KeyboardInterrupt:
            exit()
        except:
            print('retry')
    print(url+' 请求成功')
    return r

# r = get('http://www.reuters.com/article/json/data-idUSL3N1ZB0RL',headers=headers)


r = get('https://mobile.reuters.com/assets/jsonHeadlines', headers=headers, params=params)
stutas = r.status_code
# except:
datas = r.json()['headlines'] # 可以通过筛选时间筛选出前一天的数据


workbook = xlwt.Workbook(encoding = 'utf-8')
worksheet = workbook.add_sheet('reuters',cell_overwrite_ok=True)
worksheet.write(0,0,'headline')
worksheet.write(0,1,'id')
worksheet.write(0,2,'url')
worksheet.write(0,3,'dateMillis')
worksheet.write(0,4,'content')
worksheet.write(0,5,'published')


for idx in tqdm(range(0,len(datas))):
    # worksheet.write(idx+1,0,datas[idx]['headline'])
    url = article_url+datas[idx]['id']
    worksheet.write(idx+1,1,datas[idx]['id'])
    worksheet.write(idx+1,2,url)
    worksheet.write(idx+1,3,datas[idx]['dateMillis'])
    r = get(url, headers=headers)
    data = r.json()['story']
    content = data['body']
    # published = dateutil.parser.parse(str(datetime.datetime.fromtimestamp(news_json['published']).date()))
    worksheet.write(idx+1,4,content)
    worksheet.write(idx+1,0,data['headline'])
    # worksheet.write(idx+1,5,published)


workbook.save('data.xls')

# for data in tqdm(datas):
#     url = host + data['url']
#     r = requests.get(url, headers=headers,verify=False)
#     print(r.text)
    # data[-1:].

