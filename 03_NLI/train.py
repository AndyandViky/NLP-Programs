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
import pandas as pd

from pytorch_pretrained_bert import BertAdam
from dataset import get_dataloader
from config import DEVICE
from utils import tensorized
from model import Bert, Classifier
from sklearn.metrics import accuracy_score as ACC, precision_score as P, recall_score as R, f1_score as F1


train_dataloader, valid_dataloader, test_dataloader, vocab = get_dataloader(batch_size=8)
xe_loss = nn.CrossEntropyLoss().to(DEVICE)

model = nn.DataParallel(Bert().to(DEVICE))
classifier = nn.DataParallel(Classifier(xe_loss).to(DEVICE))
optim = BertAdam(model.parameters(), lr=1e-5)
c_optim = torch.optim.Adam(classifier.parameters(), lr=1e-3)

for epoch in range(10):
    model.train()
    classifier.train()
    total_loss = 0
    for i, batch in enumerate(train_dataloader):

        data, mask = tensorized(batch[:, 0], vocab)
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
    for i, batch in enumerate(valid_dataloader):

        data, mask = tensorized(batch[:, 0], vocab)
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

    print(len(preds[preds == 1]), len(labels[labels == 1]))
    acc = ACC(preds, labels)
    pre = P(preds, labels)
    rec = R(preds, labels)
    f1 = F1(preds, labels)

    print(acc, pre, rec, f1)

model.eval()
classifier.eval()
with torch.no_grad():
    preds, ids = [], []
    for i, batch in enumerate(test_dataloader):
        data, mask = tensorized(batch[:, 0], vocab)
        id = np.array(list(batch[:, 1]))
        data, mask = data.to(DEVICE), mask.to(DEVICE)
        output = model(data, mask)
        logit, loss = classifier(output)
        pred = torch.argmax(torch.softmax(logit, dim=1), dim=1).data.cpu().numpy()

        preds = np.concatenate((preds, pred))
        ids = np.concatenate((ids, id))

    preds = preds.astype(np.int)
    pd.DataFrame(np.vstack((ids, preds)).T).to_csv('./result.csv', index=False, header=['id', 'label'])