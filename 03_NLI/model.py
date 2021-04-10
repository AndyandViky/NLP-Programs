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
import torch.nn.functional as F

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
            nn.Linear(768, 2),
        )

    def forward(self, x: Tensor, label: Tensor = None) -> Tuple:

        output = self.model(x)[:, 0]
        loss = None
        if label is not None:
            loss = self.xe_loss(output, label)
        return output, loss


class FocalLoss(nn.Module):
    def __init__(self, alpha: int = 1,
                 gamma: int = 2,
                 reduce: bool = True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):

        BCE_loss = F.cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
