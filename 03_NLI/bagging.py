# -*- coding: utf-8 -*-
"""
@Author: andy
@Contact: andy_viky@163.com
@Github: https://github.com/AndyandViky
@Csdn: https://blog.csdn.net/AndyViky
@File: bagging.py
@Time: 2021/4/12 下午8:57
@Desc: bagging.py
"""
import pandas as pd

pred1 = pd.read_csv('../result_b_A.csv')
pred2 = pd.read_csv('../result_e_A.csv')
pred3 = pd.read_csv('../result_r_A.csv')
pred4 = pd.read_csv('../result_w_A.csv')

pred = pred1['label'] + pred2['label'] + pred3['label'] + pred4['label']
pred[pred < 2] = 0
pred[pred > 2] = 1
pred[pred == 2] = pred1['label'][pred == 2]

pred1['label'] = pred
pred1.to_csv('../result_A.csv', index=False)

pred1 = pd.read_csv('../result_b_B.csv')
pred2 = pd.read_csv('../result_e_B.csv')
pred3 = pd.read_csv('../result_r_B.csv')
pred4 = pd.read_csv('../result_w_B.csv')

pred = pred1['label'] + pred2['label'] + pred3['label'] + pred4['label']
pred[pred < 2] = 0
pred[pred > 2] = 1
pred[pred == 2] = pred1['label'][pred == 2]

pred1['label'] = pred

pred1.to_csv('../result_B.csv', index=False)

