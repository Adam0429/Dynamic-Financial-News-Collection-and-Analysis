# Dynamic-Financial-News-Collection-and-Analysis
Final project

## 计划

将股票涨跌改为线性的(涨跌百分比),与新闻线性情感做相关性测试,测试后五天的涨跌


按论文方法复现:从涨跌寻找相关词


## 文本情感分析可以包含几个方面：

极性分析

标签抽取（属性+评价词）

观点挖掘

观点聚类

评论主体识别

意图识别（用户需求）

评论摘要生成

主观分析

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

针对一只股票画涨跌图 情感图

SO-CAL 情感倾向计算器

## 一些实验
## 股票涨跌（当天与后五天的）与新闻情感极性的相关性（结论是新闻对涨跌是正相关的，因为当天的新闻对其股价涨跌相关度最大）
   
          scores     rates

scores  1.000000  0.050964

rates   0.050964  1.000000

          scores     rates

scores  1.000000  0.011592

rates   0.011592  1.000000


         scores    rates

scores  1.00000  0.01245

rates   0.01245  1.00000

          scores     rates

scores  1.000000 -0.009172

rates  -0.009172  1.000000

          scores     rates

scores  1.000000  0.001899

rates   0.001899  1.000000

          scores     rates

scores  1.000000  0.002559

rates   0.002559  1.000000


## 从论文中复现情感词库的生成: 

Context-aware Sentiment Detection From Ratings

Yichao Lu, Ruihai Dong, Barry Smyth


情感词库:

arguing_stock -1.0 0 3

illegal_state -1.0 0 6

sharp_percent -1.0 0 5

sharp_reporting -1.0 0 4

social_governance -1.0 0 3

serious_address -1.0 0 5

serious_bid -1.0 0 4

serious_value -1.0 0 7

serious_called -1.0 0 4

illegal_received -1.0 0 6

illegal_government -1.0 0 3

sharp_contrast -1.0 0 3

black_networks -1.0 0 5

black_programming -1.0 0 5

tough_providers -1.0 0 3

tough_facing -1.0 0 3

tough_competition -1.0 0 5

tough_industry -1.0 0 3

developing_cancer -1.0 0 4

illegal_aid -1.0 0 4

========================

arguing_hurt 1.0 6 0

arguing_consumers 1.0 6 0

nuclear_plants 1.0 3 0

institutional_pension 1.0 3 0

social_platforms 1.0 3 0

announcing_plan 1.0 3 0

providing_cash 1.0 3 0

providing_produce 1.0 3 0

institutional_declined 1.0 4 0

boosting_percent 1.0 4 0

Essential_hired 1.0 4 0

developing_health 1.0 3 0

developing_insurer 1.0 3 0

separate_meetings 1.0 3 0

social_information 1.0 4 0

separate_company 1.0 7 0

soft_supplier 1.0 4 0

soft_smartphones 1.0 4 0

announcing_sale 1.0 5 0

announcing_jumped 1.0 3 0