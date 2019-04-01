# Dynamic-Financial-News-Collection-and-Analysis
Final project

## 计划

分析单个公司的情感词典

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


## Ds词典与bing Liu的覆盖率

pos:0.2857142857142857

neg:0.18

## 从论文中复现情感词库的生成: 

Context-aware Sentiment Detection From Ratings

Yichao Lu, Ruihai Dong, Barry Smyth


情感词库:

stronger_investor 1.0 54

sharp_maker 1.0 20

stronger_inflation 1.0 19

big_season 1.0 19

stronger_shrugged 1.0 18

sharp_bank 1.0 18

strong_company 1.0 15

western_stock 1.0 14

western_target 1.0 14

optimistic_analyst 1.0 12

electric_business 1.0 10

speculative_report 1.0 10

big_sale 1.0 10

optimistic_forecast 1.0 10

big_product 1.0 10

big_firm 1.0 10

broad_index 1.0 10

foreign_tax 1.0 10

lower_bond 1.0 10

lower_expense 1.0 10


========================

lower_source -1.0 20

lower_maker -1.0 16

volatile_market -1.0 15

electric_automaker -1.0 14

lower_year -1.0 14

single_company -1.0 12

arconic_share -1.0 12

arconic_said -1.0 12

underscored_smartphone -1.0 12

interactive_software -1.0 12

public_deal -1.0 10

weak_maker -1.0 10

crucial_quarter -1.0 10

lower_insurer -1.0 10

electric_stock -1.0 10

lower_told -1.0 10

soft_supplier -1.0 10

lower_end -1.0 9

popular_company -1.0 9

broad_share -1.0 9


## 模型预测结果

预测采用10折交叉验证,并且shuffle后训练(考虑到数据集是时间上连续的，可能存在某种关系)

ds================

平均准确率： 0.6076089447410473

平均召回率： 0.41000742419772457

平均精确率： 0.41698504108280987

平均F-measure： 0.400451257266068

## 不用测试数据训练情感词典(-2000条)

准确率： 0.533

召回率： 0.3753872053872054

精确率： 0.37473770604120094

## 不用测试数据训练情感词典(-1000条)

准确率： 0.589

召回率： 0.39597025016903314

精确率： 0.39672185560862544

sn================

平均准确率： 0.5038342014641579

平均召回率： 0.34218687812857074

平均精确率： 0.3433430040257277

平均F-measure： 0.3204387236026667

bl=================

平均准确率： 0.5227797889647257

平均召回率： 0.3525505755518538

平均精确率： 0.3527937691817277

平均F-measure： 0.34509709626033774


pmi================

平均准确率： 0.551428519078233

平均召回率： 0.37660430751826734

平均精确率： 0.41407994942848336

平均F-measure： 0.33301625031038296


context==============(转化率太低)

平均准确率： 0.015441528653478512

平均召回率： 0.34154251393144597

平均精确率： 0.40251904115158876

平均F-measure： 0.018174198501710603
