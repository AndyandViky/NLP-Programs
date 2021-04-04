# -*- coding: utf-8 -*-
"""
@Author: andy
@Contact: andy_viky@163.com
@Github: https://github.com/AndyandViky
@Csdn: https://blog.csdn.net/AndyViky
@File: utils.py
@Time: 2021/4/4 下午3:34
@Desc: utils.py
"""
import torch
from typing import Tuple


def tensorized(batch: list, map: dict) -> Tuple:

    PAD = map['[PAD]']

    max_len = max([len(i) for i in batch])

    batch_tensor = torch.tensor([item + [PAD for i in range(max_len - len(item))] for item in batch])
    # batch各个元素的长度
    lengths = [len(l) for l in batch]
    mask = torch.ones_like(batch_tensor)
    for index, _ in enumerate(mask):
        mask[index, lengths[index]:] = 0

    return batch_tensor, mask