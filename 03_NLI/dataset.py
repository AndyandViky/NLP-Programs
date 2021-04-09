# -*- coding: utf-8 -*-
"""
@Author: andy
@Contact: andy_viky@163.com
@Github: https://github.com/AndyandViky
@Csdn: https://blog.csdn.net/AndyViky
@File: dataset.py
@Time: 2021/4/3 下午7:52
@Desc: dataset.py
"""
import json
import pandas as pd
import numpy as np
import torch

from torchvision import transforms
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from config import DATA_DIR, DataType, Stopwords
from pytorch_pretrained_bert import BertTokenizer


class DataUtils:

    def __init__(self):
        pass

    def prejson(self, input_path: str) -> pd.DataFrame:

        tmp = []
        try:
            content = open(input_path, 'r')
        except Exception:
            content = []
        for line in content:
            tmp.append(json.loads(line))
        data = pd.DataFrame(tmp)
        return data

    def process(self) -> Tuple:

        train_df = pd.DataFrame()
        valid_df = pd.DataFrame()
        test_df = pd.DataFrame()
        for i in ['长', '短']:
            for j in ['长', '短']:
                for p in ['A', 'B']:
                    train = self.prejson(DATA_DIR + i + j + '匹配' + p + '类/train.txt')
                    dev = self.prejson(DATA_DIR + i + j + '匹配' + p + '类/valid.txt')
                    train_df = pd.concat([train, train_df], axis=0, ignore_index=True)
                    valid_df = pd.concat([dev, valid_df], axis=0, ignore_index=True)

        train_df['labelA'] = train_df['labelA'].fillna(0).astype(int)
        train_df['labelB'] = train_df['labelB'].fillna(0).astype(int)

        valid_df['labelA'] = valid_df['labelA'].fillna(0).astype(int)
        valid_df['labelB'] = valid_df['labelB'].fillna(0).astype(int)

        train_df['label'] = train_df['labelA'] + train_df['labelB']
        valid_df['label'] = valid_df['labelA'] + valid_df['labelB']

        train_df.drop(["labelA", "labelB"], axis=1, inplace=True)
        valid_df.drop(["labelA", "labelB"], axis=1, inplace=True)

        train_data = train_df[['source', 'target', 'label']].values
        valid_data = valid_df[['source', 'target', 'label']].values

        test_file = ['短短匹配A类', '短长匹配A类', '长长匹配A类', '长长匹配B类', '短长匹配B类', '短短匹配B类']
        for f in test_file:
            test = self.prejson(DATA_DIR + f + '/test_with_id.txt')
            test_df = pd.concat([test_df, test], axis=0, ignore_index=True)
        test_data = test_df[['source', 'target', 'id']].values

        return train_data, valid_data, test_data


class MyData(Dataset):

    def __init__(self,
                 datas: np.ndarray,
                 tokenizer: BertTokenizer,
                 transform: transforms.Compose = None,
                 up_s: bool = False):
        super(MyData, self).__init__()

        self.tokenizer = tokenizer
        self.transform = transform
        self.datas = self.process(datas, up_s)

    @staticmethod
    def delete_stop_word(tokens: List[str]) -> List:

        return [item for item in tokens if item not in Stopwords]

    def process(self, datas: np.ndarray, up_s: bool = False) -> np.ndarray:

        # datas[:, 0][:1000] = [self.delete_stop_word(self.tokenizer.tokenize(i)) for i in datas[:, 0][:1000]]
        # datas[:, 1][:1000] = [self.delete_stop_word(self.tokenizer.tokenize(i)) for i in datas[:, 1][:1000]]
        #
        # return datas[:1000]

        datas[:, 0] = [self.delete_stop_word(self.tokenizer.tokenize(i)) for i in datas[:, 0]]
        datas[:, 1] = [self.delete_stop_word(self.tokenizer.tokenize(i)) for i in datas[:, 1]]

        if up_s:
            # up-sampling
            datas = np.vstack((datas, np.repeat(datas[datas[:, 2] == 1], 1, axis=0)))
        return datas

    def get_sentence_by_window(self, s: List[str], k: int, max_len: int) -> List[str]:

        l = len(s)
        if l < max_len:
            return s
        window = max_len // k
        t_point = l // k
        res = []
        i = 0
        while i < l:
            res += s[i: i+window]
            i += t_point

        return res

    def truncate_sentence(self, s1: List[str], s2: List[str]) -> List[str]:

        l1 = len(s1)
        l2 = len(s2)
        c_w = 3
        max_len = 254
        half_len = max_len // 2

        if l1 >= half_len and l2 >= half_len:
            s1, s2 = self.get_sentence_by_window(s1, c_w, half_len), self.get_sentence_by_window(s2, c_w, half_len)
        elif l1 <= half_len and l2 >= half_len:
            s2 = s2[:max_len - l1]
        elif l1 >= half_len and l2 <= half_len:
            s1 = s1[:max_len - l2]
        else:
            pass

        return ['[CLS]'] + s1 + ['[SEP]'] + s2

    def __getitem__(self, index: int) -> Tuple:

        data = self.datas[index]
        label = data[2]
        # 对于A类任务，比较宽松，不需要太长的文本；对于B类需要长文本。
        data = self.tokenizer.convert_tokens_to_ids(
            self.truncate_sentence(data[0], data[1])
        )
        if self.transform:
            data = self.transform(data)

        return data, label

    def __len__(self):
        return len(self.datas)


def get_dataloader(shuffle: bool = True, batch_size: int = 64) -> Tuple:

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_basic_tokenize=True)
    print('loading pretrained token...')
    vocab = tokenizer.vocab
    def _get_dataloder(datas: np.ndarray, shuffle: bool, up_s: bool = False) -> DataLoader:

        dataset = MyData(datas, tokenizer, up_s=up_s)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda batch: np.array(batch, dtype=object)
        )

        return dataloader

    train, valid, test = DataUtils().process()
    train_dataloader = _get_dataloder(train, shuffle, False)
    valid_dataloader = _get_dataloder(valid, shuffle)
    test_dataloader = _get_dataloder(test, False)

    return train_dataloader, valid_dataloader, test_dataloader, vocab
