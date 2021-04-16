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
import jieba

from torchvision import transforms
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from config import DATA_DIR, DataType, Stopwords
from pytorch_pretrained_bert import BertTokenizer
from gensim.summarization.summarizer import summarize


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

    def process(self, type: DataType) -> Tuple:

        train_df = pd.DataFrame()
        valid_df = pd.DataFrame()
        test_df = pd.DataFrame()
        for i in ['长', '短']:
            for j in ['长', '短']:
                for p in type.value:
                    train = self.prejson(DATA_DIR + i + j + '匹配' + p + '类/train.txt')
                    train1 = self.prejson(DATA_DIR + 'round2/' + i + j + '匹配' + p + '类.txt')
                    dev = self.prejson(DATA_DIR + i + j + '匹配' + p + '类/valid.txt')
                    train_df = pd.concat([train, train1, train_df], axis=0, ignore_index=True)
                    valid_df = pd.concat([dev, valid_df], axis=0, ignore_index=True)

        if type is DataType.A or type is DataType.B:
            l = 'label' + type.value[0]
            train_df[l] = train_df[l].fillna(0).astype(int)

            valid_df[l] = valid_df[l].fillna(0).astype(int)

            train_df['label'] = train_df[l]
            valid_df['label'] = valid_df[l]

            train_df.drop([l], axis=1, inplace=True)
            valid_df.drop([l], axis=1, inplace=True)

            test_file = [
                '短短匹配{}类'.format(type.value[0]),
                '短长匹配{}类'.format(type.value[0]),
                '长长匹配{}类'.format(type.value[0])]
            if type is DataType.B:
                test_file = test_file[::-1]
            for f in test_file:
                test = self.prejson(DATA_DIR + f + '/test_with_id.txt')
                test_df = pd.concat([test_df, test], axis=0, ignore_index=True)

        else:
            train_df['labelA'] = train_df['labelA'].fillna(0).astype(int)
            train_df['labelB'] = train_df['labelB'].fillna(0).astype(int)

            valid_df['labelA'] = valid_df['labelA'].fillna(0).astype(int)
            valid_df['labelB'] = valid_df['labelB'].fillna(0).astype(int)

            train_df['label'] = train_df['labelA'] + train_df['labelB']
            valid_df['label'] = valid_df['labelA'] + valid_df['labelB']

            train_df.drop(["labelA", "labelB"], axis=1, inplace=True)
            valid_df.drop(["labelA", "labelB"], axis=1, inplace=True)

            test_file = ['短短匹配A类', '短长匹配A类', '长长匹配A类', '长长匹配B类', '短长匹配B类', '短短匹配B类']
            for f in test_file:
                test = self.prejson(DATA_DIR + f + '/test_with_id.txt')
                test_df = pd.concat([test_df, test], axis=0, ignore_index=True)

        train_data = train_df[['source', 'target', 'label']].values
        valid_data = valid_df[['source', 'target', 'label']].values
        test_data = test_df[['source', 'target', 'id']].values

        return train_data, valid_data, test_data


class MyData(Dataset):

    def __init__(self,
                 datas: np.ndarray,
                 tokenizer: BertTokenizer,
                 type: DataType,
                 transform: transforms.Compose = None,
                 up_s: bool = False):
        super(MyData, self).__init__()

        self.tokenizer = tokenizer
        self.type = type
        self.transform = transform
        self.datas = self.process(datas, up_s)

    @staticmethod
    def delete_stop_word(tokens: List[str]) -> List:

        return [item for item in tokens if item not in Stopwords]

    @staticmethod
    def clean(content: str) -> str:
        content = content.replace('\n', '')  # 删除句子分隔符
        content = content.replace(' ', '')  # 删除空格
        return content

    def get_summary(self, s: str) -> str:

        s = s.replace(' ', '')

        if len(s) > 255:
            word_list = s.split('。')
            input_content = [' '.join(list(jieba.cut(item))) for item in word_list if item != '']
            try:
                if len(input_content) > 1:
                    summary = self.clean(''.join(summarize('。\n'.join(input_content))))
                    if summary:
                        s = summary
            except:
                pass

        return s

    def process(self, datas: np.ndarray, up_s: bool = False) -> np.ndarray:

        datas[:, 0] = [self.delete_stop_word(self.tokenizer.tokenize(i)) for i in datas[:, 0]]
        datas[:, 1] = [self.delete_stop_word(self.tokenizer.tokenize(i)) for i in datas[:, 1]]

        if up_s:
            # up-sampling
            u = np.repeat(datas[datas[:, 2] == 1], 1, axis=0)
            u[:, 0], u[:, 1] = u[:, 1], u[:, 0]
            datas = np.vstack((datas, u))
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
        if self.type is DataType.A:
            data = self.tokenizer.convert_tokens_to_ids(
                ['[CLS]'] + data[0][:127] + ['[SEP]'] + data[1][:127]
            )
        else:
            data = self.tokenizer.convert_tokens_to_ids(
                ['[CLS]'] + data[0][:127] + ['[SEP]'] + data[1][:254 - len(data[0][:127])]
            )

        if self.transform:
            data = self.transform(data)

        return data, label

    def __len__(self):
        return len(self.datas)


def get_dataloader(
        type: DataType,
        shuffle: bool = True,
        batch_size: int = 64,
) -> Tuple:

    tokenizer = BertTokenizer.from_pretrained(DATA_DIR, do_basic_tokenize=True)
    print('loading pretrained token...')
    vocab = tokenizer.vocab
    def _get_dataloder(datas: np.ndarray, shuffle: bool, up_s: bool = False) -> DataLoader:

        dataset = MyData(datas, tokenizer, up_s=up_s, type=type)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda batch: np.array(batch, dtype=object)
        )

        return dataloader

    train, valid, test = DataUtils().process(type)
    train_dataloader = _get_dataloder(train, shuffle, True)
    valid_dataloader = _get_dataloder(valid, shuffle)
    test_dataloader = _get_dataloder(test, False)

    return train_dataloader, valid_dataloader, test_dataloader, vocab
