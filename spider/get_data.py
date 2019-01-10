import requests
import socks
import socket
import time
import datetime
import time
import IPython
from tqdm import tqdm


socks.set_default_proxy(socks.SOCKS5,"127.0.0.1",1080)
socket.socket = socks.socksocket
s = requests.session()
# s.proxies = {"http": "127.0.0.1:1080","https": "127.0.0.1:1080"}
host = 'https://www.reuters.com'
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
    ('limit',500),
    ('endTime',now)
)
r = requests.get('https://www.reuters.com/assets/jsonWireNews', headers=headers, params=params)
datas = r.json()['headlines'] # 可以通过筛选时间筛选出前一天的数据
for data in tqdm(datas):
    url = host + data['url']
    r = requests.get(url, headers=headers,verify=False)
    print(r.text)
    # data[-1:].
IPython.embed()

