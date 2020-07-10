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


def get_punctuation() -> str:

    z_punctuation = '[' + string.punctuation + u' a-zA-Z0-9·！，+？＂＃＄％%＆＇（）＊★＋－／：＜＝＞＠［＼］＾＿｀｛｜｝～' \
                                               u'｟｠｢｣､〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.]'
    return z_punctuation


DATA_DIR = './datas/QA_data'
# ================================= pre-processing data ================================= #
def get_process_data(train: list) -> Tuple:

    entity = ['n_disease', 'n_crop', 'n_medicine']
    # split data and labels
    def split_raw_data(sequence: str, index: int) -> Tuple[np.ndarray, str, list]:
        sequence = sequence.replace('，', '，/dj ')
        sequence = sequence.replace('%', '%/pj ')
        sequence = sequence.replace('+', '+/aj ')
        sequence = sequence.replace('＋', '+/aj ')
        sequence = sequence.replace('·', '·/w ')
        sequence = sequence.replace('％', '%/pj ')
        raw_arr = sequence.split(' ')
        datas = np.array([i.split('/') for i in raw_arr if i != '' and len(i.split('/')) == 2])
        # delete_index = []
        for ind, item in enumerate(datas):
            item[0] = item[0].replace(' ', '')
        #     if len(item[0]) == 1:
        #         # delete single symbol
        #         if item[0] in get_punctuation():
        #             delete_index.append(ind)
        #     else:
        #         # delete symbol inner sentence
        #         item[0] = re.sub(get_punctuation(), '', item[0])
        #         if item[0] == '':
        #             delete_index.append(ind)
        # datas = np.delete(datas, delete_index, axis=0)
        # 623, 2081, 2516
        if index == 623:
            datas[26][0] = re.sub(u'[0-9]', '', datas[26][0])
        if index == 2081:
            datas[26][0] = re.sub(u'[0-9]', '', datas[26][0])
        if index == 2516:
            datas[1][0] = re.sub(u'[0-9]', '', datas[1][0])

        keys = datas[:, 1]
        values = datas[:, 0]
        values = np.delete(values, [index for index, k in enumerate(keys) if k not in entity])
        true_values = []
        for item in train[index, 1:]:
            true_values = true_values + eval(item)

        if len(values) == len(true_values):
            for i in values:
                if i not in true_values:
                    print(1)
        else:
            print(1)

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
    for ind, item in enumerate(train[:, 0]):
        word_data, seq_data, char_labels = split_raw_data(item, ind)
        train_seqs.append(seq_data)
        train_char_labels.append(char_labels)
        word_datas.append(word_data)

    return train_seqs, train_char_labels, word_datas


def build_corpus(test_seqs: list, train_seqs: list, use_word: bool = False, vector_size: int = 150) -> Tuple:
    # combine test data to build vocab
    seq_datas = train_seqs + test_seqs

    def char_split(sentence: str) -> list:
        s_arr = []
        for word in sentence:
            s_arr.append(word)
        return s_arr

    # calculate word2vector
    token = [char_split(i) for i in seq_datas]
    if use_word:
        token.insert(0, ['<pad>', '<start>', '<end>', '<unk>'])
        model = Word2Vec(token, window=10, size=vector_size, min_count=0).wv
        vocab = dict(
            [(key, value.index) for key, value in model.vocab.items()]
        )
        vectors = model.vectors
        return vocab, token[1:], vectors
    else:
        vocab = {}
        return vocab, token[1:], None


def seq2id(token: list, vocab: dict, train_char_labels: list, train_length: int, crf: bool = False) -> Tuple:

    if crf:
        train_char_labels = [['<start>'] + item + ['<end>'] for item in train_char_labels]
        token = [['<start>'] + item + ['<end>'] for item in token]
    # char to id
    def get_char_id(sequence: list, vocab: dict) -> list:
        ids = []
        for char in sequence:
            ids.append(vocab.get(char))
        return ids
    train_char_ids = np.array([get_char_id(sequence, vocab) for sequence in token[:train_length]])
    test_char_ids = np.array([get_char_id(sequence, vocab) for sequence in token[train_length:]])

    category = np.unique(np.array(sum(train_char_labels, [])))
    category_dict = dict(
        [(c, index) for index, c in enumerate(category)]
    )
    category_dict.update({'<pad>': len(category_dict)})
    category_dict.update({'<unk>': len(category_dict)})

    def get_label_id(labels: list):
        ids = []
        for c in labels:
            # label to id
            ids.append(category_dict[c])
        return ids
    train_char_labels = np.array([get_label_id(labels) for labels in train_char_labels])

    return train_char_ids, train_char_labels, test_char_ids, category_dict


# ================================= data ================================= #
class AnswerData(Dataset):
    """
    building user`s data container
    """
    def __init__(self, root: str,
                 train_char_ids: np.ndarray,
                 train_char_labels: np.ndarray,
                 test_char_ids: np.ndarray,
                 train_length: int,
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
        self.train = train
        self.fold_index = fold_index
        self.train_length = train_length
        self.train_char_ids = train_char_ids
        self.train_char_labels = train_char_labels
        self.test_char_ids = test_char_ids
        data_type = {
            'TRAIN': 0,
            'VALID': 1,
            'TEST': 2,
        }
        self.datas, self.labels = self.preprocess(data_type[data_type_name])

    def __getitem__(self, item: int) -> Tuple:

        if self.labels is not None:
            data, label = self.datas[item], self.labels[item]

            return data, label
        else:
            data = self.datas[item]

            return data

    def __len__(self) -> int:
        return len(self.datas)

    def preprocess(self, data_type: int = 0) -> Tuple:

        train_index, valid_index = get_10_fold_index(self.fold_index, self.train_length)
        if data_type == 0:
            return self.train_char_ids[train_index], self.train_char_labels[train_index]
        elif data_type == 1:
            return self.train_char_ids[valid_index], self.train_char_labels[valid_index]
        elif data_type == 2:
            return self.test_char_ids, None
        else:
            raise Exception


def get_data_loader(root: str,
                    data_type_name: list,
                    batch_size: list,
                    train_char_ids: np.ndarray,
                    train_char_labels: np.ndarray,
                    test_char_ids: np.ndarray,
                    train_length: int,
                    split_type: int = 0,
                    shuffle: list = [True, False, False],
                    fold_index: int = 0,
                    test: bool = True) -> list:

    dataloader = []
    for i in range(len(batch_size)):
        dataset = AnswerData(root,
                             data_type_name=data_type_name[i],
                             split_type=split_type,
                             fold_index=fold_index,
                             train_length=train_length,
                             train_char_ids=train_char_ids,
                             train_char_labels=train_char_labels,
                             test_char_ids=test_char_ids, )
        dataloader.append(DataLoader(dataset, shuffle=shuffle[i], batch_size=batch_size[i],
                                     collate_fn=lambda batch: np.array(batch)))

    return dataloader if test else dataloader[:2]


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

    def caculate_loss(self, output: Tensor, target: Tensor, category_dict: dict, criterion=None) -> Tensor:

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

    def get_word_id(self, output: Tensor, category_dict: dict, lengths=None) -> Tensor:

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

    def caculate_loss(self, output: Tensor, target: Tensor, category_dict: dict, criterion=None) -> Tensor:

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

    def get_word_id(self, ouput: Tensor, category_dict: dict, lengths=None) -> Tensor:

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
def drop_entity(pred: np.ndarray, test_seqs: list, category_dict: dict) -> Tuple:
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


def tensorized(batch: list, map: dict) -> Tuple:

    PAD = map['<pad>']

    max_len = max([len(i) for i in batch])

    batch_tensor = torch.tensor([item + [PAD for i in range(max_len - len(item))] for item in batch])
    # batch各个元素的长度
    lengths = [len(l) for l in batch]

    return batch_tensor, lengths


def build_lexi(data: np.ndarray) -> Tuple:

    crops = data[:, 0]
    diseases = data[:, 1]
    medicines = data[:, 2]

    l_crops = []
    for item in crops:
        item = eval(item)
        for i in item:
            if i not in l_crops:
                l_crops.append(i)

    l_diseases = []
    for item in diseases:
        item = eval(item)
        for i in item:
            if i not in l_diseases:
                l_diseases.append(i)

    l_medicines = []
    for item in medicines:
        item = eval(item)
        for i in item:
            if i not in l_medicines:
                l_medicines.append(i)
    return l_crops, l_diseases, l_medicines


def delete_symbol(entitys: list) -> list:

    pun = get_punctuation()
    for in1, item in enumerate(entitys):
        delete_index = []
        for in2, entity in enumerate(item):
            if len(entity) == 1:
                delete_index.append(in2)
            else:
                entitys[in1][in2] = re.sub(pun, '', entitys[in1][in2])
        entitys[in1] = np.delete(np.array(entitys[in1]), delete_index).tolist()
    return entitys


# ================================= main ================================= #
if __name__ == '__main__':

    train = pd.read_csv('{}/train/train.csv'.format(DATA_DIR), index_col=0).values
    test = pd.read_csv('{}/test/test.csv'.format(DATA_DIR)).values
    test_seqs = test[:, 1].tolist()
    (l_crops, l_diseases, l_medicines) = build_lexi(train[:, 1:])

    VECTOR_SIZE = 150
    pre_train = True
    USING_CRF = False
    train_seqs, train_char_labels, word_datas = get_process_data(train)
    vocab, token, vectors = build_corpus(test_seqs, train_seqs, pre_train, VECTOR_SIZE)
    train_char_ids, train_char_labels, test_char_ids, category_dict = seq2id(token, vocab, train_char_labels, len(train_seqs), USING_CRF)

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    TRAIN = 'TRAIN'
    VALID = 'VALID'
    TEST = 'TEST'
    HIDDEN_DIM = 64
    OUTPUT_SIZE = len(category_dict)
    BATCH_SIZE = 64
    EM_DIM = VECTOR_SIZE
    LR = 1e-2
    INPUT_SIZE = len(vocab)
    NUM_LAYERS = 1
    EPOCH = 300

    # init model
    model = BiLSTM(INPUT_SIZE, HIDDEN_DIM, EM_DIM, OUTPUT_SIZE, NUM_LAYERS, pre_model=torch.from_numpy(vectors)).to(
        DEVICE)
    # init optimization
    optim = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.5, 0.99))
    # init criterion
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    # get data iterator
    (train_dataloader, valid_dataloader, test_dataloader) = get_data_loader(
        root='',
        data_type_name=[TRAIN, VALID, TEST],
        batch_size=[BATCH_SIZE, 5000, 5000],
        fold_index=0,
        train_length=len(train_seqs),
        train_char_labels=train_char_labels,
        train_char_ids=train_char_ids,
        test_char_ids=test_char_ids,
    )

    valid_seqs = np.array(train_seqs)[get_10_fold_index(0, len(train_seqs))[1]]

    print('begin training ......')
    fold_index = 0  # using 10-fold cross validation
    for epoch in range(EPOCH):
        # training
        model.train()
        train_loss = 0
        if (epoch + 1) % 10 == 0:
            fold_index = (fold_index + 1) % 10
            (train_dataloader, valid_dataloader) = get_data_loader(
                root='',
                data_type_name=[TRAIN, VALID, TEST],
                batch_size=[BATCH_SIZE, 5000, 5000],
                fold_index=fold_index,
                train_length=len(train_seqs),
                train_char_labels=train_char_labels,
                train_char_ids=train_char_ids,
                test_char_ids=test_char_ids,
                test=False,
            )
            valid_seqs = np.array(train_seqs)[get_10_fold_index(fold_index, len(train_seqs))[1]]
        for index, batch in enumerate(train_dataloader):
            data, _ = tensorized(batch[:, 0], vocab)
            label, _ = tensorized(batch[:, 1], category_dict)
            data, label = data.to(DEVICE), label.to(DEVICE)

            model.zero_grad()
            optim.zero_grad()

            output = model(data)
            loss = model.caculate_loss(output, label, category_dict, criterion)

            loss.backward()
            optim.step()

            train_loss += loss.item()

        # validation
        model.eval()
        with torch.no_grad():
            batch = next(iter(valid_dataloader))
            data, lengths = tensorized(batch[:, 0], vocab)
            label, _ = tensorized(batch[:, 1], category_dict)
            data, label = data.to(DEVICE), label.to(DEVICE)
            output = model(data)
            pred = model.get_word_id(output, category_dict, lengths)
            valid_loss = model.caculate_loss(output, label, category_dict, criterion)
            p, r, f = caculate_f_acc(*drop_entity(pred, valid_seqs, category_dict),
                                     true_labels=np.array(train)[get_10_fold_index(fold_index, len(train_seqs))[1], 1:])

        print('epoch: {}, train_loss: {}, valid_loss: {}, precision: {}, recall: {}, f1: {}'.
              format(epoch, train_loss / len(train_dataloader), valid_loss, p, r, f))

    torch.save(model.state_dict(), './model.pkl')
    # model.load_state_dict(torch.load('./model.pkl', map_location=DEVICE))
    # model.eval()
    # with torch.no_grad():
    #     batch = next(iter(valid_dataloader))
    #     data, lengths = tensorized(batch[:, 0], vocab)
    #     label, _ = tensorized(batch[:, 1], category_dict)
    #     data, label = data.to(DEVICE), label.to(DEVICE)
    #     output = model(data)
    #     pred = model.get_word_id(output, category_dict, lengths)
    #     valid_loss = model.caculate_loss(output, label, category_dict, criterion)
    #     p, r, f = caculate_f_acc(*drop_entity(pred, valid_seqs, category_dict),
    #                              true_labels=np.array(train)[get_10_fold_index(fold_index, len(train_seqs))[1], 1:])

    # testing
    model.eval()
    with torch.no_grad():
        batch = next(iter(test_dataloader))
        data, lengths = tensorized(batch, vocab)
        data = data.to(DEVICE)

        output = model(data)
        pred = model.get_word_id(output, category_dict, lengths)

        crops, diseases, medicines = drop_entity(pred, test_seqs, category_dict)
        crops = delete_symbol(crops)
        diseases = delete_symbol(diseases)
        medicines = delete_symbol(medicines)
        # crops = delete_symbol(crops)
        # diseases = delete_symbol(diseases)
        # medicines = delete_symbol(medicines)
        pd.DataFrame([test[:, 0], crops, diseases, medicines]).T. \
            to_csv('./result.csv', header=['id', 'n_crop', 'n_disease', 'n_medicine'], index=False)
