# -*- coding: utf-8 -*-
"""
@Author: andy
@Contact: andy_viky@163.com
@Github: https://github.com/AndyandViky
@Csdn: https://blog.csdn.net/AndyViky
@File: LSTM_NER.py
@Time: 2020/7/3 上午9:42
@Desc: LSTM_NER.py
"""
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader

DATA_DIR = './datas/QA_data'
'''
get data
'''
train = pd.read_csv('{}/train/train.csv'.format(DATA_DIR)).values
test = pd.read_csv('{}/test/test.csv'.format(DATA_DIR)).values


class AnswerData(Dataset):
    """
    building user`s data container
    """
    def __init__(self, root, data_name, transform, train=True):
        super(AnswerData, self).__init__()

        self.root = root
        self.data_name = data_name
        self.transform = transform
        self.train = train

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass

    def preprocess(self):
        pass
print(1)