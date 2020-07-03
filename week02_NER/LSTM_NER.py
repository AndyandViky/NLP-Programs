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
import re
import pandas as pd
import numpy as np
import string

from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from gensim.models import Word2Vec

z_punctuation = '[' + string.punctuation + u' a-zA-Z0-9·！？。＂＃＄％%＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.]'
DATA_DIR = './datas/QA_data'
'''
pre-processing data
'''
train = pd.read_csv('{}/train/train.csv'.format(DATA_DIR), index_col=0).values
test = pd.read_csv('{}/test/test.csv'.format(DATA_DIR)).values


# split data and labels
def split_raw_data(sequence: str) -> Tuple[np.ndarray, str]:
    sequence = sequence.replace('，', '')
    sequence = sequence.replace('//', '/')
    raw_arr = sequence.split(' ')
    datas = np.array([i.split('/') for i in raw_arr if i != '' and len(i.split('/')) == 2])
    # delete symbol
    delete_index = []
    for ind, item in enumerate(datas):
        item[0] = item[0].replace(' ', '')
        if len(item[0]) == 1:
            # delete single symbol
            if item[0] in z_punctuation:
                delete_index.append(ind)
        else:
            # delete symbol inner sentence
            item[0] = re.sub(z_punctuation, '', item[0])
    datas = np.delete(datas, delete_index, axis=0)

    seq_data = ''
    for j in datas[:, 0]:
        seq_data = seq_data + j
    return datas, seq_data


train_seqs = []
for item in train[:, 0]:
    _, seq_data = split_raw_data(item)
    train_seqs.append(seq_data)
# combine test data to build vocab
test_seqs = [re.sub(z_punctuation, '', sentence) for sentence in test[:, 1]]
seq_datas = train_seqs + test_seqs


def char_split(sentence: str) -> list:
    s_arr = []
    for word in sentence:
        s_arr.append(word)
    return s_arr


# calculate word2vector
token = [char_split(i) for i in seq_datas]
model = Word2Vec(token, window=5, size=200).wv


class AnswerData(Dataset):
    """
    building user`s data container
    """
    def __init__(self, root: str, data_name: str, transform, train: bool = True, type: int = 0):
        """
        :param root:
        :param data_name:
        :param transform:
        :param train:
        :param type: 0 represent using char, 1: using word, 2: using Lattice
        """
        super(AnswerData, self).__init__()

        self.root = root
        self.data_name = data_name
        self.transform = transform
        self.train = train

    def __getitem__(self, item: int) -> Tensor:
        pass

    def __len__(self) -> int:
        pass

    def preprocess(self) -> Tensor:
        pass
print(1)