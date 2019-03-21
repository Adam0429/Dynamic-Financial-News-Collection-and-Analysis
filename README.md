# Dynamic-Financial-News-Collection-and-Analysis
Final project

## 计划

将股票涨跌改为线性的(涨跌百分比),与新闻线性情感做相关性测试,测试后五天的涨跌


按论文方法复现:从涨跌寻找相关词


## 爬虫数据来源

Reuters 

Booloomberg


股票涨幅做label  新闻正文-->提到的股票公司-->股票统计涨跌-->涨1,跌0


正文出现的公司代码，公司名称 去对应股票


后期考虑加入:


知识图谱 Wiki Data ，法人，竞争公司，公司标签等属性


各种属性分配不同的权重


attention mechanism

对于特定的股票也许有特有的feature(如法人，竞争公司，公司标签等属性)。统计出来看一下

## 一些实验
## 股票涨跌（当天与后五天的）与新闻情感极性的相关性（结论是新闻对涨跌是正相关的，因为当天的新闻对其股价涨跌相关度最大）
   
          scores     rates

scores  1.000000  0.047048

rates   0.047048  1.000000

          scores     rates

scores  1.000000  0.010678

rates   0.010678  1.000000

          scores     rates

scores  1.000000  0.013753

rates   0.013753  1.000000

          scores     rates

scores  1.000000 -0.008817

rates  -0.008817  1.000000

          scores     rates

scores  1.000000  0.007262

rates   0.007262  1.000000

          scores     rates

scores  1.000000 -0.003498

rates  -0.003498  1.000000


## 从论文中复现情感词库的生成: 

Context-aware Sentiment Detection From Ratings

Yichao Lu, Ruihai Dong, Barry Smyth


情感词库:

rompting_profit -1.0
prompting_economy -1.0                                     
representing_inc -1.0
bad_news -0.36636639172885904
ongoing_part -6.879569489615561e
accusing_lawsuit -6.879569489615561e
giving_month -6.879569489615561e-05
worried_impact -6.879569489615561e-05
ongoing_trade 0.17789533430063653
leaving_company 0.2437923769669283
representing_group 0.4382878138462364
sexual_misconduct 0.6901976005690359
older_pickup 1.0
tracking_mobile 1.0
tracking_technology 1.0
accusing_pharmaceutical 1.0
winning_day 1.0
older_weakness 1.0
ongoing_progress 1.0
turning_netflix 1.0
========================
prompting_profit -1.0
prompting_economy -1.0
representing_inc -1.0
bad_news -0.36636639172885904
ongoing_part -6.879569489615561e-05
accusing_lawsuit -6.879569489615561e-05
giving_month -6.879569489615561e-05
worried_impact -6.879569489615561e-05
ongoing_trade 0.17789533430063653
leaving_company 0.2437923769669283
representing_group 0.4382878138462364
sexual_misconduct 0.6901976005690359
older_pickup 1.0
tracking_mobile 1.0
tracking_technology 1.0
accusing_pharmaceutical 1.0
winning_day 1.0
older_weakness 1.0
ongoing_progress 1.0
turning_netflix 1.0
