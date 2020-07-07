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
import torch
import torch.nn as nn
import pandas as pd

from torch.nn import LSTM, GRU
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from gensim.models import Word2Vec

from utils.utils import caculate_accuracy

z_punctuation = '[' + string.punctuation + u' a-zA-Z0-9·！？。＂＃＄％%＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.]'
DATA_DIR = './datas/QA_data'
'''
pre-processing data
'''
train = pd.read_csv('{}/train/train.csv'.format(DATA_DIR), index_col=0).values
test = pd.read_csv('{}/test/test.csv'.format(DATA_DIR)).values

entity = ['n_disease', 'n_crop', 'n_medicine']
# split data and labels
def split_raw_data(sequence: str) -> Tuple[np.ndarray, str, list]:
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

    char_labels = []
    for (word, label) in datas:
        if label not in entity:
            char_labels = char_labels + ['O' for i in range(len(word))]
        else:
            char_labels.append('B_{}'.format(label))
            char_labels = char_labels + ['I_{}'.format(label) for i in range(len(word) - 1)]

    seq_data = ''
    for char in datas[:, 0]:
        seq_data = seq_data + char
    return datas, seq_data, char_labels


train_seqs = []
train_char_labels = []
word_datas = []
for item in train[:, 0]:
    word_data, seq_data, char_labels = split_raw_data(item)
    train_seqs.append(seq_data)
    train_char_labels.append(char_labels)
    word_datas.append(word_data)

# combine test data to build vocab
test_seqs = [re.sub(z_punctuation, '', sentence) for sentence in test[:, 1]]
seq_datas = train_seqs + test_seqs


def char_split(sentence: str) -> list:
    s_arr = []
    for word in sentence:
        s_arr.append(word)
    return s_arr
# calculate word2vector
VECTOR_SIZE = 100
token = [char_split(i) for i in seq_datas]
token.insert(0, ['UNK'])
model = Word2Vec(token, window=10, size=VECTOR_SIZE, min_count=0).wv
vocab = model.vocab
vectors = model.vectors
max_seq_len = max([len(i) for i in token])


# calculate char id
def get_char_id(seqence):
    ids = []
    for char in seqence:
        ids.append(vocab[char].index)
    return ids
train_char_ids = [get_char_id(seqence) for seqence in train_seqs]
test_char_ids = [get_char_id(seqence) for seqence in test_seqs]


# pad data and labels
UNK_INDEX = vocab['UNK'].index
train_char_ids = [item + [UNK_INDEX for i in range(max_seq_len - len(item))] for item in train_char_ids]
train_char_ids = np.array(train_char_ids)
test_char_ids = [item + [UNK_INDEX for i in range(max_seq_len - len(item))] for item in test_char_ids]
test_char_ids = np.array(test_char_ids)

train_char_labels = [item + ['UNK' for i in range(max_seq_len - len(item))] for item in train_char_labels]
category = np.unique(np.array(sum(train_char_labels, [])))
category_dict = dict(
    [(c, index) for index, c in enumerate(category)]
)
train_char_labels = np.array(train_char_labels)
for c in category:
    train_char_labels[train_char_labels == c] = category_dict[c]
train_char_labels = train_char_labels.astype(np.int)

VALID_PART = 1 / 9
valid_seqs = train_seqs[int(len(train_seqs) * (1 - VALID_PART)):]


class AnswerData(Dataset):
    """
    building user`s data container
    """
    def __init__(self, root: str, transform, data_type_name: str = 'TRAIN', split_type: int = 0):
        """
        :param root:
        :param data_name:
        :param transform:
        :param train:
        :param type: 0 represent using char, 1: using word, 2: using Lattice
        """
        super(AnswerData, self).__init__()

        self.root = root
        self.transform = transform
        self.train = train
        data_type = {
            'TRAIN': 0,
            'VALID': 1,
            'TEST': 2,
        }
        self.datas, self.labels = self.preprocess(data_type[data_type_name])

    def __getitem__(self, item: int) -> Tuple:

        if self.labels is not None:
            data, label = self.datas[item], self.labels[item]
            data = self.transform(data)

            return data, torch.tensor(label)
        else:
            data = self.datas[item]
            data = self.transform(data)

            return data

    def __len__(self) -> int:
        return len(self.datas)

    def preprocess(self, data_type: int = 0) -> Tuple:

        train_index = int(len(train_seqs) * (1 - VALID_PART))
        if data_type == 0:
            return train_char_ids[:train_index], train_char_labels[:train_index]
        elif data_type == 1:
            return train_char_ids[train_index:], train_char_labels[train_index:]
        elif data_type == 2:
            return test_char_ids, None
        else:
            raise Exception


def get_data_loader(root: str, data_type_name: str, split_type: int, batch_size: int, shuffle: bool = True) -> DataLoader:

    dataset = AnswerData(root, transform=lambda x: torch.tensor(x), data_type_name=data_type_name, split_type=split_type)
    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)

    return dataloader


class Model(nn.Module):

    def __init__(self, input_dim: int,
                 hidden_dim: int = 64,
                 em_dim: int = 50,
                 output_dim: int = 10,
                 num_layers: int = 1,
                 bidirectional: bool = True,
                 pre_model: Tensor = None):
        super(Model, self).__init__()

        self.bidirectional = bidirectional
        if pre_model is not None:
            self.embedding = nn.Embedding.from_pretrained(pre_model)
        else:
            self.embedding = nn.Embedding(input_dim, em_dim)
        self.rnn = LSTM(
            input_size=em_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.out = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x: Tensor) -> Tensor:

        embeded = self.embedding(x)
        output, hidden = self.rnn(embeded)
        output = torch.softmax(self.out(output), dim=2)
        return output


def drop_entity(pred: np.ndarray, test_seqs: list) -> Tuple:
    b_crop = category_dict['B_n_crop']
    i_crop = category_dict['I_n_crop']
    b_disease = category_dict['B_n_disease']
    i_disease = category_dict['I_n_disease']
    b_medicine = category_dict['B_n_medicine']
    i_medicine = category_dict['I_n_medicine']

    def get_entity_id(seq_item: list, item: np.ndarray, b_e, i_e):
        begin = np.where(item == b_e)[0]
        result = []
        if len(begin) == 0:
            return result
        for index, j in enumerate(begin):
            if j >= len(seq_item):
                break
            seq = ''
            seq = seq + seq_item[j]
            if j != len(item) - 1:
                for k in range(j + 1, len(item)):
                    if item[k] == i_e:
                        seq = seq + seq_item[k]
                    else:
                        break
            result.append(seq)
        return result

    crops = []
    diseases = []
    medicines = []
    for index, item in enumerate(pred):
        crops.append(get_entity_id(test_seqs[index], item, b_crop, i_crop))
        diseases.append(get_entity_id(test_seqs[index], item, b_disease, i_disease))
        medicines.append(get_entity_id(test_seqs[index], item, b_medicine, i_medicine))

    return crops, diseases, medicines


def caculate_f_acc(crops: list, diseases: list, medicines: list) -> Tuple:

    precision = 0
    recall = 0
    f_scores = 0
    n = len(crops)
    train_index = int(len(train_seqs) * (1 - VALID_PART))
    true_labels = train[train_index:, 1:]

    def calculate_tl_fl(item: list, true_item: list) -> Tuple:
        true_item = eval(true_item)
        ttl = 0
        for j in item:
            if j in true_item:
                ttl = ttl + 1
        return ttl, len(item), len(true_item)

    for i in range(n):
        ttl, size, t_size = calculate_tl_fl(crops[i], true_labels[i, 0])
        result = calculate_tl_fl(diseases[i], true_labels[i, 1])
        ttl = ttl + result[0]
        size = size + result[1]
        t_size = t_size + result[2]
        result = calculate_tl_fl(medicines[i], true_labels[i, 2])
        ttl = ttl + result[0]
        size = size + result[1]
        t_size = t_size + result[2]
        p = ttl / size if size != 0 else 0
        r = ttl / t_size if t_size != 0 else 0
        f1 = (2 * p * r) / (p + r) if (p + r) != 0 else 0
        f_scores = f_scores + f1
        precision = precision + p
        recall = recall + r

    return precision / n, recall / n, f_scores / n


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
TRAIN = 'TRAIN'
VALID = 'VALID'
TEST = 'TEST'
HIDDEN_DIM = 64
OUTPUT_SIZE = len(category)
BATCH_SIZE = 64
EM_DIM = VECTOR_SIZE
LR = 1e-2
INPUT_SIZE = max_seq_len
NUM_LAYERS = 1
EPOCH = 150

# init model
model = Model(INPUT_SIZE, HIDDEN_DIM, EM_DIM, OUTPUT_SIZE, NUM_LAYERS, pre_model=torch.from_numpy(vectors)).to(DEVICE)
# init optimization
optim = torch.optim.Adam(model.parameters(), lr=LR)
# init criterion
criterion = nn.CrossEntropyLoss().to(DEVICE)
# get_dataloader
train_dataloader = get_data_loader(root='', data_type_name=TRAIN, split_type=0, batch_size=BATCH_SIZE)
valid_dataloader = get_data_loader(root='', data_type_name=VALID, split_type=0, batch_size=5000, shuffle=False)
test_dataloader = get_data_loader(root='', data_type_name=TEST, split_type=0, batch_size=5000, shuffle=False)

print('begin training ......')
for epoch in range(EPOCH):
    # training
    model.train()
    train_loss = 0
    for index, (data, label) in enumerate(train_dataloader):
        data, label = data.to(DEVICE), label.to(DEVICE)

        model.zero_grad()
        optim.zero_grad()

        output = model(data)
        loss = criterion(output.view((-1, output.size(2))), label.view(-1))

        loss.backward()
        optim.step()

        train_loss += loss.item()

    # validation
    model.eval()
    with torch.no_grad():
        data, label = next(iter(valid_dataloader))
        data, label = data.to(DEVICE), label.to(DEVICE)
        output = model(data)
        pred = torch.argmax(output, 2).data.cpu().numpy()
        valid_loss = criterion(output.view((-1, output.size(2))), label.view(-1))
        valid_acc = caculate_accuracy(torch.argmax(output, 2).view(-1).data.cpu(),
                                      label.view(-1).data.cpu())
        p, r, f = caculate_f_acc(*drop_entity(pred, valid_seqs))

    print('epoch: {}, train_loss: {}, valid_loss: {}, acc: {}, precision: {}, recall: {}, f1: {}'.
          format(epoch, train_loss / len(train_dataloader), valid_loss, valid_acc, p, r, f))

# torch.save(model.state_dict(), './model.pkl')
# model.load_state_dict(torch.load('./model.pkl', map_location=DEVICE))
# model.eval()
# with torch.no_grad():
#     data, label = next(iter(valid_dataloader))
#     data, label = data.to(DEVICE), label.to(DEVICE)
#     output = model(data)
#     pred = torch.argmax(output, 2).data.cpu().numpy()
#     valid_loss = criterion(output.view((-1, output.size(2))), label.view(-1))
#     valid_acc = caculate_accuracy(torch.argmax(output, 2).view(-1).data.cpu(),
#                                   label.view(-1).data.cpu())
#     valid_f_acc = caculate_f_acc(*drop_entity(pred, valid_seqs))
#
# print('valid_loss: {}, acc: {}, f_acc: {}'.format(valid_loss, valid_acc, valid_f_acc))

# testing
model.eval()
with torch.no_grad():
    data = next(iter(test_dataloader))
    data = data.to(DEVICE)

    output = model(data)
    pred = torch.argmax(output, 2).data.cpu().numpy()

    crops, diseases, medicines = drop_entity(pred, test_seqs)
    pd.DataFrame([test[:, 0], crops, diseases, medicines]).T.\
        to_csv('./result.csv', header=['id', 'n_crop', 'n_disease', 'n_medicine'], index=False)
