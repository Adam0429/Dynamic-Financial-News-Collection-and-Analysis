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
         
        scores    rates
               
scores  1.00000  0.01073

rates   0.01073  1.00000

        scores     rates
        
scores  1.000000 -0.007373

rates  -0.007373  1.000000

          scores     rates

scores  1.000000  0.019477

rates   0.019477  1.000000

         scores     rates

scores  1.000000 -0.003658

rates  -0.003658  1.000000

         scores     rates

scores  1.000000 -0.075723

rates  -0.075723  1.000000

## 从论文中复现情感词库的生成: 

Context-aware Sentiment Detection From Ratings

Yichao Lu, Ruihai Dong, Barry Smyth


情感词库:

stock_stocks -49389.001322313336

stock_indexes -197.56534472913475

stock_companies -197.56534472913475

said_strategy -30.582629436215125

said_u -22.28163273039045

said_percent -22.28163273039045

investment_e -13.991643535427317

create_e -13.991643535427317

create_fund -13.991643535427317

asset_banks -4.252466078618018

========================

new_inc 13.865215462366422

stock_pct 19.280970135917414

investment_german 30.158519970948976

create_german 30.158519970948976

stock_live 58.52612564857034

stock_profit 58.52612564857034

stock_updates 58.52612564857034

new_live 58.52612564857034

investment_fund 190.52059246630097

share_inc 190.52059246630097
