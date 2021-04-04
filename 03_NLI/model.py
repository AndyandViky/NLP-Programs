# -*- coding: utf-8 -*-
"""
@Author: andy
@Contact: andy_viky@163.com
@Github: https://github.com/AndyandViky
@Csdn: https://blog.csdn.net/AndyViky
@File: model.py
@Time: 2021/4/4 下午3:31
@Desc: model.py
"""
import torch.nn as nn
import torch

from typing import Tuple
from torch import Tensor
from config import BERT_DIR
from pytorch_pretrained_bert import BertModel as bert


class Bert(nn.Module):

    def __init__(self):
        super(Bert, self).__init__()

        self.model = bert.from_pretrained(BERT_DIR)

    def forward(self, x: Tensor, attention_mask=None) -> Tensor:

        return self.model(x, attention_mask=attention_mask, output_all_encoded_layers=False)[0]


class Classifier(nn.Module):
    def __init__(self, xe_loss):
        super(Classifier, self).__init__()

        self.xe_loss = xe_loss
        self.model = nn.Sequential(
            nn.Linear(768, 2)
        )

    def forward(self, x: Tensor, label: Tensor) -> Tuple:

        output = self.model(x)[:, 0]
        loss = self.xe_loss(output, label)
        return output, loss
