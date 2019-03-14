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

live_day -552500.9382821129

soft_estimates -552500.9382821129

wide_apple -552500.9382821129

red_trading -12403.93645483703

old_david -4200.42240470336

red_quarter -2277.837353015036

soft_software -1954.1826934248666

red_apple -1509.9421143199709

red_tax -259.1937636289998

old_morgan -233.949354378491

warning_demand -117.52539385002163

old_paypal -117.52539385002163

red_rights -117.52539385002163

red_month -117.52539385002163

red_support -117.52539385002163

soft_google -117.52539385002163

old_banking -117.52539385002163

able_products -117.52539385002163

soft_profit -117.52539385002163

old_filing -117.52539385002163

========================

red_years 910.5891523059694

red_time 910.5891523059694

old_technologies 910.5891523059694

old_citigroup 2751.7052428746124

red_end 2751.7052428746306

live_deliver 2751.7052428746306

red_street 2751.7052428746306

soft_results 2751.7052428746306

old_news 2751.7052428746306

old_amazon 2751.7052428746306

old_outlook 2751.7052428746306

old_motors 2751.7052428746306

red_north 2751.7052428746306

wide_profit 2751.7052428746306

red_summit 2751.7052428746306

easing_shares 2751.7052428746306

easing_quarter 2751.7052428746306

red_rate 2751.7052428746306

old_charges 2751.7052428746306

old_cash 16314.745407323479
