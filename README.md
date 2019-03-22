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
针对一只股票画涨跌图 情感图

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

issing_inc -1.0 0 3

worst_caterpillar -1.0 0 3

worst_day -1.0 0 4

representing_inc -1.0 0 3

least_percent -1.0 0 6

least_apple -1.0 0 3

worst_year -1.0 0 6

monetary_chair -1.0 0 4

seeing_impact -1.0 0 3

reducing_company -1.0 0 3

worst_percent -1.0 0 4

least_glass -1.0 0 4

least_time -1.0 0 4

pending_billion -1.0 0 4

hard_hit -0.44778509041787956 1 5

programming_warner -0.3634664400516885 1 4

least_year -0.3634664400516885 1 4

developing_cancer -0.3634664400516885 1 4

competitive_company -0.2533900216455929 1 3

hard_technology -0.2533900216455929 1 3

least_million -0.2533900216455929 1 3

programming_viacom -0.1137986915882441 1 2

missing_analyst -0.1137986915882441 1 2

hard_bank -0.1137986915882441 1 2

programming_rival -0.1137986915882441 1 2

institutional_michael -0.1137986915882441 1 2

institutional_director -0.1137986915882441 1 2

institutional_trading -0.1137986915882441 1 2

worst_week -0.1137986915882441 1 2

hard_caterpillar -0.1137986915882441 1 2

missing_quarter -0.1137986915882441 1 2

least_one -0.1137986915882441 1 2

monetary_decision -0.1137986915882441 1 2

reducing_stake -0.1137986915882441 1 2

missing_billion -0.06463188611941212 3 5

reducing_inc -0.021690552433151118 3 4

least_billion -2.0385030322496052e-05 3 3

hard_maker -2.0385030322495605e-05 2 2

missing_percent 0.011374144641696145 5 4

monetary_reserve 0.011374144641696145 5 4

========================
original_content 1.0 17 0

institutional_morgan 1.0 5 0

institutional_stanley 1.0 5 0

original_cash 1.0 3 0

original_risk 1.0 3 0

original_netflix 1.0 13 0

original_year 1.0 9 0

original_carbon 1.0 3 0

competitive_year 1.0 3 0

original_iphone 1.0 4 0

original_maker 1.0 4 0

representing_volkswagen 1.0 3 0

original_summit 1.0 5 0

original_ltd 1.0 16 0

original_owner 1.0 5 0

waning_iphone 1.0 9 0

waning_face 1.0 8 0

waning_billion 1.0 7 0

original_subscriber 1.0 4 0

original_apple 1.0 4 0

original_ruling 1.0 3 0

developing_stage 1.0 4 0

developing_health 1.0 3 0

developing_insurer 1.0 3 0

developing_acquisition 1.0 4 0

competitive_online 1.0 3 0

competitive_offer 1.0 3 0

developing_electric 1.0 3 0

computing_intel 1.0 3 0

institutional_pension 1.0 3 0

competing_share 1.0 3 0

original_part 1.0 3 0

original_spending 1.0 3 0

original_marketing 1.0 3 0

original_billion 1.0 4 0

representing_tech 1.0 4 0

original_series 1.0 3 0

competing_number 1.0 4 0

original_drug 1.0 3 0

reducing_comment 1.0 3 0
