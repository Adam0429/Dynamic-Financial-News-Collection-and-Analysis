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

serious_policies -1.0 0 3

blaming_percent -1.0 0 3

blaming_warning -1.0 0 3

announcing_companies -1.0 0 3

serious_address -1.0 0 8

serious_called -1.0 0 4

serious_bid -1.0 0 4

serious_value -1.0 0 7

renewable_tax -1.0 0 4

serious_said -1.0 0 3

single_companies -1.0 0 3

Mexican_control -1.0 0 5

serious_see -1.0 0 3

serious_proposal -1.0 0 3

continuing_euros -1.0 0 4

Cable_providers -0.40810495496362637 1 4

Cable_facing -0.40810495496362637 1 4

serious_take -0.2978788357652759 1 3

commercial_supplier -0.15066206959874162 1 2

commercial_aerospace -0.15066206959874162 1 2

========================

announcing_plan 1.0 4 0

continuing_profit 1.0 3 0

commercial_businesses 1.0 5 0

huge_percent 1.0 3 0

commercial_leader 1.0 3 0

huge_said 1.0 3 0

Southern_said 1.0 3 0

single_mid 1.0 5 0

single_revenue 1.0 4 0

commercial_rules 1.0 3 0

commercial_relations 1.0 3 0

commercial_announced 1.0 3 0

announcing_jumped 1.0 3 0

commercial_maker 1.0 6 0

Belgian_metals 1.0 4 0

commercial_ride 1.0 4 0

commercial_automaker 1.0 3 0

commercial_sharing 1.0 3 0

Mexican_said 1.0 3 0

Belgian_company 1.0 18 0
