# Text Matching Baseline

## 简介
----
分别对任务 A 和任务 B 进行建模，之后进行结果合并。  
master 分支：一个基于原始 bert 的 Baseline，线上分数 0.72 左右。  
berts 分支：roberta-wwm-ext; bert-wwm-ext; ernie-1.0 Baseline，模型融合之后精度有一定上涨。

## 训练
测试环境：python 3.8; pytorch 1.7.1; pytorch-pretrained-bert 0.6.2; 两张 2080ti 一轮大概 20 分钟，5 轮之内收敛。

----
###
Tips: baseline.py 文件为苏神提供的模型 https://github.com/bojone/sohu2021-baseline