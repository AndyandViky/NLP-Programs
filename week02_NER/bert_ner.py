# -*- coding: utf-8 -*-
'''
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: bert_ner.py
@time: 2020/9/3 16:38
@desc: bert_ner.py
'''

import pandas as pd
import torch
import numpy as np
import re
import torch.nn as nn
import torch
from torch.optim import Adam

from pytorch_pretrained_bert import BertTokenizer, BertModel as bert, BertAdam
from week02_NER.Bi_LSTM_NER import get_process_data, get_data_loader, caculate_f_acc, drop_entity, get_10_fold_index
from typing import Tuple
from torch import Tensor
# ================================= pre-processing data ================================= #
# ================================= pre-processing data ================================= #


# ================================= model ================================= #
class BertModel(nn.Module):

    def __init__(self):
        super(BertModel, self).__init__()

        self.model = bert.from_pretrained('./datas/bert-base-chinese/')

    def forward(self, x, attention_mask=None) -> Tensor:

        return self.model(x, attention_mask=attention_mask, output_all_encoded_layers=False)[0]

    # def get_word_id(self, output: Tensor) -> Tensor:
    #     output = torch.softmax(output, 2)
    #     return torch.argmax(output, 2).data.cpu().numpy()


class Classifier(nn.Module):
    def __init__(self, category_dict: dict, criterion):
        super(Classifier, self).__init__()

        n_cluster = len(category_dict)
        self.criterion = criterion
        self.category_dict = category_dict
        self.model = nn.Sequential(
            nn.Linear(768, n_cluster),
            nn.Softmax()
        )

    def forward(self, x, labels=None) -> Tuple:

        output = self.model(x)
        pred = self.get_word_id(output)
        if labels is None:
            return pred, None
        else:
            PAD = self.category_dict['[PAD]']
            assert PAD is not None

            mask = (labels != PAD)  # [B, L]
            targets = labels[mask]
            out_size = output.size(2)
            logits = output.masked_select(
                mask.unsqueeze(2).expand(-1, -1, out_size)
            ).contiguous().view(-1, out_size)

            assert logits.size(0) == targets.size(0)
            loss = self.criterion(logits, targets)
            return pred, loss

    def get_word_id(self, output: Tensor) -> Tensor:
        # output = torch.softmax(output, 2)
        return torch.argmax(output, 2)
# ================================= model ================================= #


def label2id(train_labels: list) -> Tuple:

    # char to id
    def get_char_id(sequence: list, vocab: dict) -> list:
        ids = []
        for char in sequence:
            ids.append(vocab.get(char))
        return ids

    category = np.unique(np.array(sum(train_labels, [])))
    category_dict = dict(
        [(c, index) for index, c in enumerate(category)]
    )
    # category_dict.update({
    #     '[PAD]': len(category_dict) - 1
    # })

    train_labels = np.array([get_char_id(labels, category_dict) for labels in train_labels])

    return train_labels, category_dict


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


def main():

    DATA_DIR = './datas/QA_data'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    USING_CRF = False
    ENHANCE_DATA = False
    TEST_LENGTH = 500
    BATCH_SIZE = 14
    fold_index = 9

    train1 = pd.read_csv('{}/train/train.csv'.format(DATA_DIR), index_col=0).values
    train = pd.read_csv('{}/train/train1.csv'.format(DATA_DIR), index_col=0).values

    train = np.vstack((train1, train))
    _, train_seqs, _, train_char_labels = get_process_data(train, ENHANCE_DATA)
    train_char_labels = [['[PAD]'] + item + ['[PAD]'] for item in train_char_labels]
    if ENHANCE_DATA: train = np.repeat(train, 2, axis=0)
    TRAIN_LENGTH = len(train_seqs) - TEST_LENGTH
    valid_seqs = np.array(train_seqs)[get_10_fold_index(fold_index, TRAIN_LENGTH)[1]]

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_basic_tokenize=True)
    vocab = tokenizer.vocab
    tokenized_text = [tokenizer.tokenize(i) for i in train_seqs]
    # for index, i in enumerate(tokenized_text):
    #     for j in i:
    #         if len(j) > 1 and j != '[UNK]':
    #             t = train_seqs[index]
    #             print(1)
    tokenized_text = [['[CLS]'] + item + ['[SEP]'] for item in tokenized_text]
    train_ids = np.array([tokenizer.convert_tokens_to_ids(item) for item in tokenized_text])
    train_label_ids, category_dict = label2id(train_char_labels)

    test_ids = train_ids[TRAIN_LENGTH:]
    test_labels = train_label_ids[TRAIN_LENGTH:]
    train_labels = train_label_ids[:TRAIN_LENGTH]
    train_ids = train_ids[:TRAIN_LENGTH]

    xe_loss = nn.CrossEntropyLoss()
    xe_loss.to(DEVICE)
    model = BertModel().to(DEVICE)
    model = nn.DataParallel(model)
    classifier = Classifier(category_dict, xe_loss).to(DEVICE)
    classifier = nn.DataParallel(classifier)
    optim = BertAdam(model.parameters(), lr=1e-5)
    c_optim = Adam(classifier.parameters(), lr=1e-3)

    (train_dataloader, valid_dataloader, test_dataloader) = get_data_loader(
        root='',
        data_type_name=['TRAIN', 'VALID', 'TEST'],
        batch_size=[BATCH_SIZE, 5000, 5000],
        fold_index=fold_index,
        train_length=TRAIN_LENGTH,
        train_char_labels=train_labels,
        train_char_ids=train_ids,
        test_char_ids=test_ids,
    )
    # training
    best_model = None
    best_score = 0
    for epoch in range(100):
        model.train()
        train_loss = 0
        for index, batch in enumerate(train_dataloader):
            data, lengths = tensorized(batch[:, 0], vocab)
            labels, _ = tensorized(batch[:, 1], category_dict)
            data, labels = data.to(DEVICE), labels.to(DEVICE)

            model.zero_grad()
            optim.zero_grad()
            output = model(data, lengths)
            _, loss = classifier(output, labels)
            loss = loss.mean()
            loss.backward()
            optim.step()
            c_optim.step()

            train_loss += loss.item()

        # valid
        model.eval()
        with torch.no_grad():
            batch = next(iter(valid_dataloader))
            data, lengths = tensorized(batch[:, 0], vocab)
            labels, _ = tensorized(batch[:, 1], category_dict)
            data, labels = data.to(DEVICE), labels.to(DEVICE)

            output = model(data, lengths)
            pred, valid_loss = classifier(output, labels)
            pred = pred[:, 1:].data.cpu().numpy()
            p, r, f, f_acc = caculate_f_acc(*drop_entity(pred, valid_seqs, category_dict),
                                            true_labels=np.array(train)[get_10_fold_index(fold_index, TRAIN_LENGTH)[1],
                                                        1:])

            if best_score <= f_acc:
                best_score = f_acc
                best_model = model.state_dict()
        print(
            'epoch: {}, train_loss: {:.4f}, valid_loss: {:.4f}, f_acc: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1_score: {:.3f}'.
            format(epoch, train_loss / len(train_dataloader), valid_loss.item(), f_acc, p, r, f))

    print('best_score: {:.3f}'.format(best_score))
    torch.save(best_model, './model.pkl')
    model.load_state_dict(torch.load('./model.pkl', map_location=DEVICE))
    # testing
    model.eval()
    with torch.no_grad():
        data, lengths = tensorized(test_ids, vocab)
        labels, _ = tensorized(test_labels, category_dict)
        data, labels = data.to(DEVICE), labels.to(DEVICE)

        output = model(data, lengths)
        pred, _ = classifier(output)
        pred = pred[:, 1:].data.cpu().numpy()
        p, r, f, f_acc = caculate_f_acc(*drop_entity(pred, train_seqs[TRAIN_LENGTH:], category_dict),
                                        true_labels=np.array(train)[TRAIN_LENGTH:, 1:])
        # print(confusion_matrix(label.view(-1).data.cpu().numpy(), pred.reshape(-1)))
        # print(classification_report(label.view(-1).data.cpu().numpy(), pred.reshape(-1), target_names=list(category_dict.keys())[:11]))
        print('tset: f_acc: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1_score: {:.3f}'.format(f_acc, p, r, f))

    # work
    test = pd.read_csv('{}/test/test1.csv'.format(DATA_DIR)).values
    test_seqs = test[:, 1]
    test_seqs = np.array([re.sub(r'[0-9]', '0', sequence) for sequence in test_seqs])
    model.eval()
    with torch.no_grad():
        batch = next(iter(test_dataloader))
        data, lengths = tensorized(batch, vocab)
        data = data.to(DEVICE)

        output = model(data, lengths)
        pred, _ = classifier(output)
        pred = pred.data.cpu().numpy()
        crops, diseases, medicines = drop_entity(pred, test_seqs, category_dict)
        pd.DataFrame([test[:, 0], crops, diseases, medicines]).T. \
            to_csv('./result.csv', header=['id', 'n_crop', 'n_disease', 'n_medicine'], index=False)


if __name__ == '__main__':
    main()

