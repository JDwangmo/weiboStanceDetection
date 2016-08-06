# Weibo Stance Detection —— Data Set
### [微博立场分析](http://tcci.ccf.org.cn/conference/2016/pages/page05_evadata.html)

###目录
> [NLPCC_2016_Stance_Detection_gold](https://github.com/JDwangmo/weiboStanceDetection/tree/master/version_2.0/dataset#nlpcc_2016_stance_detection_gold):该项目的数据都来自于此


##### NLPCC_2016_Stance_Detection_gold：
- describe：官方提供测试数据的gold result，不过只有1000条有结果，其他14000无。
- NLPCC_2016_Stance_Detection_Task_A_gold.utf8：任务A的测试数据 1000 条 gold result，已转为utf8 编码,也手工去除部分乱码问题。
- NLPCC2016_Stance_Detection_Task_A_Result.txt: 比赛时提交的结果，有15000条，这个结果的F1宏平均是66.7%。

##### 结果汇总
- baseline_result.csv：将比赛时提交结果跟1000条gold result 比较，得到的baseline，跟官方给的结果是一致，在这1000条数据上，对了629条，F1值分别为：0.7027027，0.63040791，0.12903226;F1值宏平均（Favor和Against）为 0.666555306852。