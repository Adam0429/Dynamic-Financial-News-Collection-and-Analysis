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

worst_day -1.0 0 6

worst_year -1.0 0 6

illegal_state -1.0 0 6

sharp_percent -1.0 0 5

sharp_reporting -1.0 0 4

worst_percent -1.0 0 3

worst_suffered -1.0 0 4

social_governance -1.0 0 3

major_ga -1.0 0 5

illegal_received -1.0 0 6

illegal_government -1.0 0 3

sharp_contrast -1.0 0 3

difficult_make -1.0 0 3

black_network -1.0 0 5

black_programming -1.0 0 5

troubled_pipeline -1.0 0 3

troubled_expansion -1.0 0 3

troubled_billion -1.0 0 3

tough_provider -1.0 0 3

tough_facing -1.0 0 3

tough_competition -1.0 0 5

tough_industry -1.0 0 3

illegal_billion -1.0 0 3

major_economy -1.0 0 3

illegal_aid -1.0 0 4

organic_sale -1.0 0 3

sharp_drop -0.5501693380477118 1 6

extra_million -0.4046193381680032 1 4

major_business -0.4046193381680032 1 4

tough_face -0.2943508006148632 1 3


========================

major_restructuring 1.0 3 0

major_automaker 1.0 5 0

social_platform 1.0 4 0

major_representing 1.0 4 0

major_tech 1.0 4 0

whole_lot 1.0 3 0

major_chipmaker 1.0 3 0

regular_percent 1.0 8 0

regular_close 1.0 4 0

regular_trading 1.0 4 0

institutional_declined 1.0 4 0

great_citing 1.0 3 0

great_good 1.0 4 0

great_iphone 1.0 4 0

major_expectation 1.0 3 0

whole_bank 1.0 3 0

electronic_simplified 1.0 14 0

separate_meeting 1.0 3 0

major_moved 1.0 3 0

separate_company 1.0 8 0

major_administration 1.0 4 0

major_took 1.0 3 0

major_issue 1.0 3 0

social_information 1.0 4 0

soft_supplier 1.0 4 0

external_finance 1.0 3 0

major_airline 1.0 3 0

optimistic_analyst 1.0 4 0

optimistic_subscriber 1.0 4 0

renewable_network 1.0 10 0