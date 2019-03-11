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


## 一些实验
## 股票涨跌（后五天的）与新闻情感极性的相关性
   
             scores     rates

scores  1.000000  0.031765

rates   0.031765  1.000000

          scores     rates

scores  1.000000  0.017745

rates   0.017745  1.000000

          scores     rates

scores  1.000000  0.019571

rates   0.019571  1.000000

          scores     rates

scores  1.000000  0.023022

rates   0.023022  1.000000

          scores     rates

scores  1.000000  0.013875

rates   0.013875  1.000000


## 从论文中复现情感词库的生成: 

Context-aware Sentiment Detection From Ratings

Yichao Lu, Ruihai Dong, Barry Smyth


情感词库:

inc_since -207627.74023083402

high_u -207627.74023083402

market_live -207627.7402308294

market_take -207627.7402308294

industrial_gain -207627.7402308294

industrial_day -207627.7402308294

stocks_among -207627.7402308294

led_stage -207627.7402308294

corp_growth -207627.7402308294

inc_stake -207627.7402308294

========================

stocks_p 699.5460327621929

inc_including 699.5460327621929

high_reuters 699.5460327621929

market_morgan 1106.2689809927922

corp_corporate 1106.2689809927922

main_reuters 1106.2689809927922

point_stanley 1106.2689809927922

stocks_u 1106.2689809927936

market_said 2254.183527774366

updates_pct 12929.046683319246
