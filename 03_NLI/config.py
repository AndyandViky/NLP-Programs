# -*- coding: utf-8 -*-
"""
@Author: andy
@Contact: andy_viky@163.com
@Github: https://github.com/AndyandViky
@Csdn: https://blog.csdn.net/AndyViky
@File: config.py
@Time: 2021/4/3 下午7:39
@Desc: config.py
"""
import os
import torch

from enum import Enum
from utils import load_stopwords_from_file

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'datas/sohu/')
BERT_DIR = os.path.join(ROOT_DIR, 'datas/bert-base-chinese/')

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Stopwords = load_stopwords_from_file(DATA_DIR + 'stop_words.txt')


class DataType(Enum):

    A = ['A']
    B = ['B']
    ALL = ['A', 'B']


class Args(Enum):

    mini_batch_size = 32
    batch_size = 256
    bert_lr = 1e-5
    c_lr = 1e-3
    epochs = 10
    accumulation_steps = batch_size // mini_batch_size
    type = DataType.B

    alpha = 0.25
    gamma = 3

