# -*- coding: utf-8 -*-
"""
@Author: andy
@Contact: andy_viky@163.com
@Github: https://github.com/AndyandViky
@Csdn: https://blog.csdn.net/AndyViky
@File: utils.py
@Time: 2020/6/28 下午4:23
@Desc: utils.py
"""


def caculate_accuracy(pred, target):

    correct = 0
    correct += (pred == target).sum()

    acc = correct.numpy() / len(pred.numpy())
    return acc