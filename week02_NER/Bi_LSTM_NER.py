# -*- coding: utf-8 -*-
"""
@Author: andy
@Contact: andy_viky@163.com
@Github: https://github.com/AndyandViky
@Csdn: https://blog.csdn.net/AndyViky
@File: Bi_LSTM_NER.py
@Time: 2020/7/3 上午9:42
@Desc: Bi_LSTM_NER.py
"""
import re
import numpy as np
import string
import torch
import torch.nn as nn
import pandas as pd

from torch.nn import LSTM
from itertools import zip_longest
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
max_seq_len = max([len(i) for i in seq_datas]) + 2

def char_split(sentence: str) -> list:
    s_arr = ['<start>']
    for word in sentence:
        s_arr.append(word)
    s_arr.append('<end>')
    s_arr = s_arr + ['<pad>' for i in range(max_seq_len - len(s_arr))]
    return s_arr
# calculate word2vector
VECTOR_SIZE = 150
token = [char_split(i) for i in seq_datas]
model = Word2Vec(token, window=10, size=VECTOR_SIZE, min_count=0).wv
vocab = model.vocab
vectors = model.vectors


# calculate char id
def get_char_id(seqence: list) -> list:
    ids = []
    for char in seqence:
        ids.append(vocab[char].index)
    return ids
train_char_ids = np.array([get_char_id(seqence) for seqence in token[:len(train_seqs)]])
test_char_ids = np.array([get_char_id(seqence) for seqence in token[len(train_seqs):]])


# padding labels
train_char_labels = [['<start>'] + item + ['<end>'] + ['<pad>' for i in range(max_seq_len - 2 - len(item))] for item in train_char_labels]
category = np.unique(np.array(sum(train_char_labels, [])))
category_dict = dict(
    [(c, index) for index, c in enumerate(category)]
)
train_char_labels = np.array(train_char_labels)
for c in category:
    train_char_labels[train_char_labels == c] = category_dict[c]
train_char_labels = train_char_labels.astype(np.int)


# ================================= data ================================= #
class AnswerData(Dataset):
    """
    building user`s data container
    """
    def __init__(self, root: str,
                 transform,
                 data_type_name: str = 'TRAIN',
                 split_type: int = 0,
                 fold_index: int = 0):
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
        self.fold_index = fold_index
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

        train_index, valid_index = get_10_fold_index(self.fold_index, len(train_seqs))
        if data_type == 0:
            return train_char_ids[train_index], train_char_labels[train_index]
        elif data_type == 1:
            return train_char_ids[valid_index], train_char_labels[valid_index]
        elif data_type == 2:
            return test_char_ids, None
        else:
            raise Exception


def get_data_loader(root: str,
                    data_type_name: str,
                    split_type: int,
                    batch_size: int,
                    shuffle: bool = True,
                    fold_index: int = 0) -> DataLoader:

    dataset = AnswerData(root,
                         transform=lambda x: torch.tensor(x),
                         data_type_name=data_type_name,
                         split_type=split_type,
                         fold_index=fold_index)
    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)

    return dataloader


# ================================= model ================================= #
class BiLSTM(nn.Module):

    def __init__(self, input_dim: int,
                 hidden_dim: int = 64,
                 em_dim: int = 50,
                 output_dim: int = 10,
                 num_layers: int = 1,
                 bidirectional: bool = True,
                 pre_model: Tensor = None):
        super(BiLSTM, self).__init__()

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

    def caculate_loss(self, output: Tensor, target: Tensor, criterion=None) -> Tensor:

        PAD = category_dict['<pad>']
        assert PAD is not None

        mask = (target != PAD)  # [B, L]
        targets = target[mask]
        out_size = output.size(2)
        logits = output.masked_select(
            mask.unsqueeze(2).expand(-1, -1, out_size)
        ).contiguous().view(-1, out_size)

        assert logits.size(0) == targets.size(0)

        return criterion(logits, targets)

    def get_word_id(self, output: Tensor, lengths=None) -> Tensor:

        return torch.argmax(output, 2).data.cpu().numpy()


class BiLSTM_CRF(nn.Module):

    def __init__(self, input_dim: int,
                 hidden_dim: int = 64,
                 em_dim: int = 50,
                 output_dim: int = 10,
                 num_layers: int = 1,
                 bidirectional: bool = True,
                 pre_model: Tensor = None):
        super(BiLSTM_CRF, self).__init__()

        self.bidirectional = bidirectional
        self.output_dim = output_dim
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
        self.transition = nn.Parameter(
            torch.ones(output_dim, output_dim) * 1 / output_dim)

    def forward(self, x: Tensor) -> Tensor:

        embeded = self.embedding(x)
        output, hidden = self.rnn(embeded)
        output = self.out(output)
        crf_scores = output.unsqueeze(
            2).expand(-1, -1, self.output_dim, -1) + self.transition.unsqueeze(0)

        return crf_scores

    def indexed(self, targets: Tensor, tagset_size: int, start_id: int) -> Tensor:
        """将targets中的数转化为在[T*T]大小序列中的索引,T是标注的种类"""
        batch_size, max_len = targets.size()
        for col in range(max_len - 1, 0, -1):
            targets[:, col] += (targets[:, col - 1] * tagset_size)
        targets[:, 0] += (start_id * tagset_size)
        return targets

    def caculate_loss(self, output: Tensor, target: Tensor, criterion=None) -> Tensor:

        pad_id = category_dict['<pad>']
        start_id = category_dict['<start>']
        end_id = category_dict['<end>']

        device = output.device

        # targets:[B, L] crf_scores:[B, L, T, T]
        batch_size, max_len = target.size()
        target_size = len(category_dict)

        # mask = 1 - ((targets == pad_id) + (targets == end_id))  # [B, L]
        mask = (target != pad_id)
        lengths = mask.sum(dim=1)
        targets = self.indexed(target, target_size, start_id)

        # # 计算Golden scores方法１
        # import pdb
        # pdb.set_trace()
        targets = targets.masked_select(mask)  # [real_L]

        flatten_scores = output.masked_select(
            mask.view(batch_size, max_len, 1, 1).expand_as(output)
        ).view(-1, target_size * target_size).contiguous()

        golden_scores = flatten_scores.gather(
            dim=1, index=targets.unsqueeze(1)).sum()

        # 计算golden_scores方法２：利用pack_padded_sequence函数
        # targets[targets == end_id] = pad_id
        # scores_at_targets = torch.gather(
        #     crf_scores.view(batch_size, max_len, -1), 2, targets.unsqueeze(2)).squeeze(2)
        # scores_at_targets, _ = pack_padded_sequence(
        #     scores_at_targets, lengths-1, batch_first=True
        # )
        # golden_scores = scores_at_targets.sum()

        # 计算all path scores
        # scores_upto_t[i, j]表示第i个句子的第t个词被标注为j标记的所有t时刻事前的所有子路径的分数之和
        scores_upto_t = torch.zeros(batch_size, target_size).to(device)
        for t in range(max_len):
            # 当前时刻 有效的batch_size（因为有些序列比较短)
            batch_size_t = (lengths > t).sum().item()
            if t == 0:
                scores_upto_t[:batch_size_t] = output[:batch_size_t,
                                               t, start_id, :]
            else:
                # We add scores at current timestep to scores accumulated up to previous
                # timestep, and log-sum-exp Remember, the cur_tag of the previous
                # timestep is the prev_tag of this timestep
                # So, broadcast prev. timestep's cur_tag scores
                # along cur. timestep's cur_tag dimension
                scores_upto_t[:batch_size_t] = torch.logsumexp(
                    output[:batch_size_t, t, :, :] +
                    scores_upto_t[:batch_size_t].unsqueeze(2),
                    dim=1
                )
        all_path_scores = scores_upto_t[:, end_id].sum()

        # 训练大约两个epoch loss变成负数，从数学的角度上来说，loss = -logP
        loss = (all_path_scores - golden_scores) / batch_size
        return loss

    def get_word_id(self, ouput: Tensor, lengths=None) -> Tensor:

        """使用维特比算法进行解码"""
        start_id = category_dict['<start>']
        end_id = category_dict['<end>']
        pad = category_dict['<pad>']
        tagset_size = len(category_dict)

        device = ouput.device
        # B:batch_size, L:max_len, T:target set size
        B, L, T, _ = ouput.size()
        # viterbi[i, j, k]表示第i个句子，第j个字对应第k个标记的最大分数
        viterbi = torch.zeros(B, L, T).to(device)
        # backpointer[i, j, k]表示第i个句子，第j个字对应第k个标记时前一个标记的id，用于回溯
        backpointer = (torch.zeros(B, L, T).long() * end_id).to(device)
        lengths = torch.LongTensor(lengths).to(device)
        # 向前递推
        for step in range(L):
            batch_size_t = (lengths > step).sum().item()
            if step == 0:
                # 第一个字它的前一个标记只能是start_id
                viterbi[:batch_size_t, step,
                :] = ouput[: batch_size_t, step, start_id, :]
                backpointer[: batch_size_t, step, :] = start_id
            else:
                max_scores, prev_tags = torch.max(
                    viterbi[:batch_size_t, step - 1, :].unsqueeze(2) +
                    ouput[:batch_size_t, step, :, :],  # [B, T, T]
                    dim=1
                )
                viterbi[:batch_size_t, step, :] = max_scores
                backpointer[:batch_size_t, step, :] = prev_tags

        # 在回溯的时候我们只需要用到backpointer矩阵
        backpointer = backpointer.view(B, -1)  # [B, L * T]
        tagids = []  # 存放结果
        tags_t = None
        for step in range(L - 1, 0, -1):
            batch_size_t = (lengths > step).sum().item()
            if step == L - 1:
                index = torch.ones(batch_size_t).long() * (step * tagset_size)
                index = index.to(device)
                index += end_id
            else:
                prev_batch_size_t = len(tags_t)

                new_in_batch = torch.LongTensor(
                    [end_id] * (batch_size_t - prev_batch_size_t)).to(device)
                offset = torch.cat(
                    [tags_t, new_in_batch],
                    dim=0
                )  # 这个offset实际上就是前一时刻的
                index = torch.ones(batch_size_t).long() * (step * tagset_size)
                index = index.to(device)
                index += offset.long()

            try:
                tags_t = backpointer[:batch_size_t].gather(
                    dim=1, index=index.unsqueeze(1).long())
            except RuntimeError:
                import pdb
                pdb.set_trace()
            tags_t = tags_t.squeeze(1)
            tagids.append(tags_t.tolist())

        # tagids:[L-1]（L-1是因为扣去了end_token),大小的liebiao
        # 其中列表内的元素是该batch在该时刻的标记
        # 下面修正其顺序，并将维度转换为 [B, L]
        tagids = list(zip_longest(*reversed(tagids), fillvalue=pad))
        tagids = torch.Tensor(tagids).long()

        # 返回解码的结果
        return tagids


# ================================= util functions ================================= #
def drop_entity(pred: np.ndarray, test_seqs: list) -> Tuple:
    b_crop = category_dict['B_n_crop']
    i_crop = category_dict['I_n_crop']
    b_disease = category_dict['B_n_disease']
    i_disease = category_dict['I_n_disease']
    b_medicine = category_dict['B_n_medicine']
    i_medicine = category_dict['I_n_medicine']

    def get_entity_id(seq_item: list, item: np.ndarray, b_e, i_e):
        item = item[1:]
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
                    if item[k] == i_e and k < len(seq_item):
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


def caculate_f_acc(crops: list, diseases: list, medicines: list, true_labels: np.ndarray) -> Tuple:

    precision = 0
    recall = 0
    f_scores = 0
    n = len(crops)

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


def get_10_fold_index(fold_index: int, n: int) -> Tuple:

    index = np.array([i for i in range(n)])
    valid_len = n // 10
    valid_index = index[fold_index * valid_len:(fold_index + 1) * valid_len]
    train_index = np.delete(index, valid_index)
    return train_index, valid_index


def tensorized(datas: Tensor, labels=None) -> Tuple:

    datas = datas.data.numpy()
    PAD = vocab['<pad>'].index
    C_PAD = category_dict['<pad>']

    mask = (datas != PAD)  # [B, L]
    datas = [i[mask[index]] for index, i in enumerate(datas)]
    lengths = [len(l) for l in datas]
    max_len = max([len(i) for i in datas])

    datas = [np.concatenate((item, np.array([PAD for i in range(max_len - len(item))]))) for item in datas]
    datas = torch.tensor(datas).long()
    if labels is not None:
        labels = labels.data.numpy()
        labels = [i[mask[index]] for index, i in enumerate(labels)]
        labels = [np.concatenate((item, np.array([C_PAD for i in range(max_len - len(item))]))) for item in labels]
        labels = torch.tensor(labels).long()

    return datas, labels, lengths


# ================================= main ================================= #
def main():
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    TRAIN = 'TRAIN'
    VALID = 'VALID'
    TEST = 'TEST'
    HIDDEN_DIM = 64
    OUTPUT_SIZE = len(category)
    BATCH_SIZE = 64
    EM_DIM = VECTOR_SIZE
    LR = 1e-2
    INPUT_SIZE = len(vocab)
    NUM_LAYERS = 1
    EPOCH = 150

    # init model
    model = BiLSTM(INPUT_SIZE, HIDDEN_DIM, EM_DIM, OUTPUT_SIZE, NUM_LAYERS, pre_model=torch.from_numpy(vectors)).to(
        DEVICE)
    # init optimization
    optim = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.5, 0.99))
    # init criterion
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    # get data iterator
    test_dataloader = get_data_loader(root='', data_type_name=TEST, split_type=0, batch_size=5000, shuffle=False)
    train_dataloader = get_data_loader(root='', data_type_name=TRAIN, split_type=0, batch_size=BATCH_SIZE,
                                       fold_index=0)
    valid_dataloader = get_data_loader(root='', data_type_name=VALID, split_type=0, batch_size=5000, shuffle=False,
                                       fold_index=0)
    valid_seqs = np.array(train_seqs)[get_10_fold_index(0, len(train_seqs))[1]]

    print('begin training ......')
    fold_index = 0  # using 10-fold cross validation
    for epoch in range(EPOCH):
        # training
        model.train()
        train_loss = 0
        if (epoch + 1) % 10 == 0:
            fold_index = (fold_index + 1) % 10
            train_dataloader = get_data_loader(root='', data_type_name=TRAIN, split_type=0, batch_size=BATCH_SIZE,
                                               fold_index=fold_index)
            valid_dataloader = get_data_loader(root='', data_type_name=VALID, split_type=0, batch_size=5000,
                                               shuffle=False,
                                               fold_index=fold_index)
            valid_seqs = np.array(train_seqs)[get_10_fold_index(fold_index, len(train_seqs))[1]]
        for index, (data, label) in enumerate(train_dataloader):
            data, label, _ = tensorized(data, label)
            data, label = data.to(DEVICE), label.to(DEVICE)

            model.zero_grad()
            optim.zero_grad()

            output = model(data)
            loss = model.caculate_loss(output, label, criterion)

            loss.backward()
            optim.step()

            train_loss += loss.item()

        # validation
        model.eval()
        with torch.no_grad():
            data, label = next(iter(valid_dataloader))
            data, label, lengths = tensorized(data, label)
            data, label = data.to(DEVICE), label.to(DEVICE)
            output = model(data)
            pred = model.get_word_id(output, lengths)
            valid_loss = model.caculate_loss(output, label, criterion)
            p, r, f = caculate_f_acc(*drop_entity(pred, valid_seqs),
                                     true_labels=np.array(train)[get_10_fold_index(fold_index, len(train_seqs))[1], 1:])

        print('epoch: {}, train_loss: {}, valid_loss: {}, precision: {}, recall: {}, f1: {}'.
              format(epoch, train_loss / len(train_dataloader), valid_loss, p, r, f))

    torch.save(model.state_dict(), './model.pkl')
    # model.load_state_dict(torch.load('./model.pkl', map_location=DEVICE))

    # testing
    model.eval()
    with torch.no_grad():
        data = next(iter(test_dataloader))
        data, _, lengths = tensorized(data)
        data = data.to(DEVICE)

        output = model(data)
        pred = model.get_word_id(output, lengths)

        crops, diseases, medicines = drop_entity(pred, test_seqs)
        pd.DataFrame([test[:, 0], crops, diseases, medicines]).T. \
            to_csv('./result.csv', header=['id', 'n_crop', 'n_disease', 'n_medicine'], index=False)


if __name__ == '__main__':
    main()
