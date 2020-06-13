# -*- coding: utf-8 -*-
"""
@Author: andy
@Contact: andy_viky@163.com
@Github: https://github.com/AndyandViky
@Csdn: https://blog.csdn.net/AndyViky
@File: config.py
@Time: 2020/6/9 下午10:57
@Desc: config.py
"""
import os
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Local directory of CypherCat API
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Local directory containing entire repo
ROOT_DIR = os.path.split(ROOT_DIR)[0]

# Local directory for datas
DATASETS_DIR = os.path.join(ROOT_DIR, 'datas')

