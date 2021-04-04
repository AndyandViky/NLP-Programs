# -*- coding: utf-8 -*-
"""
@Author: andy
@Contact: andy_viky@163.com
@Github: https://github.com/AndyandViky
@Csdn: https://blog.csdn.net/AndyViky
@File: train.py
@Time: 2021/4/3 上午10:23
@Desc: train.py
"""
import torch
import torch.nn as nn
import numpy as np

from pytorch_pretrained_bert import BertAdam
from dataset import get_dataloader
from config import DEVICE
from utils import tensorized
from model import Bert, Classifier
from sklearn.metrics import accuracy_score as ACC, precision_score as P, recall_score as R, f1_score as F1


train_dataloader, valid_dataloader, test_dataloader = get_dataloader(batch_size=8)
xe_loss = nn.CrossEntropyLoss().to(DEVICE)

model = nn.DataParallel(Bert().to(DEVICE))
classifier = nn.DataParallel(Classifier(xe_loss).to(DEVICE))
optim = BertAdam(model.parameters(), lr=1e-5)
c_optim = torch.optim.Adam(classifier.parameters(), lr=1e-3)

for epoch in range(5):
    model.train()
    classifier.train()
    total_loss = 0
    for i, batch in enumerate(train_dataloader[0]):

        data, mask = tensorized(batch[:, 0], train_dataloader[1])
        label = torch.tensor(list(batch[:, 1])).to(DEVICE)
        data, mask = data.to(DEVICE), mask.to(DEVICE)
        output = model(data, mask)
        logit, loss = classifier(output, label)

        loss = loss.mean()
        optim.zero_grad()
        c_optim.zero_grad()
        loss.backward()
        optim.step()
        c_optim.step()

        total_loss += loss.item()

    model.eval()
    classifier.eval()
    valid_loss = 0
    preds, labels = [], []
    for i, batch in enumerate(valid_dataloader[0]):

        data, mask = tensorized(batch[:, 0], valid_dataloader[1])
        label = torch.tensor(list(batch[:, 1])).to(DEVICE)
        data, mask = data.to(DEVICE), mask.to(DEVICE)
        output = model(data, mask)
        logit, loss = classifier(output, label)
        pred = torch.argmax(torch.softmax(logit, dim=1), dim=1).data.cpu().numpy()
        label = label.data.cpu().numpy()

        preds = np.concatenate((pred, preds))
        labels = np.concatenate((label, labels))
        loss = loss.mean()

        valid_loss += loss.item()

    print(len(preds[preds == 1]))
    acc = ACC(preds, labels)
    pre = P(preds, labels)
    rec = R(preds, labels)

    print(acc, pre, rec)