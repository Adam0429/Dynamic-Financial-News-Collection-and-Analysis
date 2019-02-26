# Dynamic-Financial-News-Collection-and-Analysis
Final project

## 问题

要服务器


1. 数据怎么处理。每句话当成一条？数据应该每个公司分一条吗？


2. 人都预测不准，模型怎么才叫准，负面新闻不一定代表股票会降。给例子


3. 预测全是一个结果怎么回事


4. 怎么调参

## 爬虫数据来源
Reuters 

Booloomberg


股票涨幅做label  新闻正文-->提到的股票公司-->股票统计涨跌-->涨1,跌0


正文出现的公司代码，公司名称 去对应股票


后期考虑加入:


知识图谱 Wiki Data ，法人，竞争公司，公司标签等属性


各种属性分配不同的权重


attention mechanism