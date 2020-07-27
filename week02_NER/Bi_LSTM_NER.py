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
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from xpinyin import Pinyin
from gensim.models import Word2Vec
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


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
                char_labels = char_labels + ['I_{}'.format(label) for i in range(len(word) - 2)]
                char_labels.append('E_{}'.format(label))

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
        model = Word2Vec(token, window=15, size=vector_size, min_count=0).wv
        vocab = dict(
            [(key, value.index) for key, value in model.vocab.items()]
        )
        vectors = model.vectors
        return vocab, token[1:], vectors, model.vocab
    else:
        char_words = []
        for seq in seq_datas:
            for char in seq:
                if char not in char_words:
                    char_words.append(char)
        vocab = dict(
            [(char, index) for index, char in enumerate(char_words)]
        )
        vocab.update({'<pad>': len(vocab)})
        vocab.update({'<unk>': len(vocab)})
        vocab.update({'<start>': len(vocab)})
        vocab.update({'<end>': len(vocab)})
        return vocab, token, [], []


def seq2id(token: list, vocab: dict, train_char_labels: list, train_length: int, crf: bool = False) -> Tuple:

    if crf:
        train_char_labels = [item + ['<end>'] for item in train_char_labels]
        token = [item + ['<end>'] for item in token]
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
    if crf:
        category_dict.update({'<start>': len(category_dict)})

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
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: Tensor, lengths: list) -> Tensor:

        embeded = self.dropout(self.embedding(x))
        packed = pack_padded_sequence(embeded, lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.rnn(packed)
        output, hidden = pad_packed_sequence(output, batch_first=True)
        output = self.out(output)
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

    def forward(self, x: Tensor, lengths: list) -> Tensor:

        embeded = self.embedding(x)
        packed = pack_padded_sequence(embeded, lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.rnn(packed)
        output, hidden = pad_packed_sequence(output, batch_first=True)
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
def drop_entity(pred: np.ndarray, test_seqs: list, category_dict: dict, post_process) -> Tuple:
    b_crop = category_dict['B_n_crop']
    i_crop = category_dict['I_n_crop']
    e_crop = category_dict['E_n_crop']
    b_disease = category_dict['B_n_disease']
    i_disease = category_dict['I_n_disease']
    e_disease = category_dict['E_n_disease']
    b_medicine = category_dict['B_n_medicine']
    i_medicine = category_dict['I_n_medicine']
    e_medicine = category_dict['E_n_medicine']

    def get_entity_id(seq_item: list, item: np.ndarray, b_e, i_e, e_e):
        begin = np.where(item == b_e)[0]
        result = []
        if len(begin) == 0:
            return result
        for index, j in enumerate(begin):
            if j >= len(seq_item):
                break
            seq = ''
            seq = seq + seq_item[j]
            last_index = 0
            if j != len(item) - 1:
                for k in range(j + 1, len(seq_item)):
                    if item[k] == i_e:
                        seq = seq + seq_item[k]
                    elif item[k] == e_e:
                        seq = seq + seq_item[k]
                        last_index = k
                        break
                    else:
                        # 扩展一位，尽可能纠错
                        if item[k + 1] == e_e and k + 1 < len(seq_item):
                            seq = seq + seq_item[k]
                            seq = seq + seq_item[k + 1]
                            last_index = k + 1
                            break
                        elif item[k + 1] == i_e and k + 1 < len(seq_item):
                            seq = seq + seq_item[k]
                        else:
                            break
            if item[last_index] == e_e:
                result.append(seq)
        return result

    crops = []
    diseases = []
    medicines = []
    for index, item in enumerate(pred):
        crops.append(get_entity_id(test_seqs[index], item, b_crop, i_crop, e_crop))
        diseases.append(get_entity_id(test_seqs[index], item, b_disease, i_disease, e_disease))
        medicines.append(get_entity_id(test_seqs[index], item, b_medicine, i_medicine, e_medicine))

    return post_process(crops, 0), post_process(diseases, 1), post_process(medicines, 2)


def caculate_f_acc(crops: list, diseases: list, medicines: list, true_labels: np.ndarray) -> Tuple:

    precision = 0
    recall = 0
    f_scores = 0
    f_accs = 0
    n = len(crops)

    def calculate_tl_fl(item: list, true_item: list) -> Tuple:
        true_item = eval(true_item)
        tl = 0
        fl = 0
        for j in item:
            if j in true_item:
                tl = tl + 1
            else:
                fl = fl + 1
        fl = fl + len(true_item) - tl
        return tl, fl, len(item), len(true_item)

    for i in range(n):
        tl, fl, size, t_size = calculate_tl_fl(crops[i], true_labels[i, 0])
        result = calculate_tl_fl(diseases[i], true_labels[i, 1])
        tl = tl + result[0]
        fl = fl + result[1]
        size = size + result[2]
        t_size = t_size + result[3]
        result = calculate_tl_fl(medicines[i], true_labels[i, 2])
        tl = tl + result[0]
        fl = fl + result[1]
        size = size + result[2]
        t_size = t_size + result[3]
        p = tl / size if size != 0 else 0
        r = tl / t_size if t_size != 0 else 0
        f1 = (2 * p * r) / (p + r) if (p + r) != 0 else 0
        f_acc = tl / (tl + fl)
        f_accs = f_accs + f_acc
        f_scores = f_scores + f1
        precision = precision + p
        recall = recall + r

    return precision / n, recall / n, f_scores / n, f_accs / n


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
    l_crops_c = []
    for item in crops:
        item = eval(item)
        for i in item:
            if i not in l_crops:
                l_crops.append(i)
                l_crops_c.append(1)
            else:
                index = l_crops.index(i)
                l_crops_c[index] += 1

    l_diseases = []
    l_diseases_c = []
    for item in diseases:
        item = eval(item)
        for i in item:
            if i not in l_diseases:
                l_diseases.append(i)
                l_diseases_c.append(1)
            else:
                index = l_diseases.index(i)
                l_diseases_c[index] += 1

    l_medicines = []
    l_medicines_c = []
    for item in medicines:
        item = eval(item)
        for i in item:
            if i not in l_medicines:
                l_medicines.append(i)
                l_medicines_c.append(1)
            else:
                index = l_medicines.index(i)
                l_medicines_c[index] += 1

    l_crops = dict(
        [(key, value) for key, value in zip(l_crops, l_crops_c)]
    )
    l_diseases = dict(
        [(key, value) for key, value in zip(l_diseases, l_diseases_c)]
    )
    l_medicines = dict(
        [(key, value) for key, value in zip(l_medicines, l_medicines_c)]
    )
    return l_crops, l_diseases, l_medicines


def build_word(word_datas: np.ndarray) -> Tuple:

    words = []
    for item in word_datas:
        word = item[:, 0]
        words.append(word.tolist())

    model = Word2Vec(words, size=128, window=10, min_count=0).wv
    word_vector = model.vectors
    vocab = model.vocab

    return word_vector, vocab


def build_word_pinyin(word_datas: np.ndarray) -> Tuple:

    pinyin = Pinyin()
    words = []
    for item in word_datas:
        word = item[:, 0]
        py = [pinyin.get_pinyin(p, '') for p in word.tolist()]
        words.append(py)
    model = Word2Vec(words, size=128, window=10, min_count=0).wv
    pinyin_vector = model.vectors
    pinyin_vocab = model.vocab

    return pinyin_vector, pinyin_vocab


def change_lexi2dict(lexi: Tuple, ex_vocab: dict, ex_vectors: np.ndarray, vocab: dict, vectors: np.ndarray) -> dict:

    lexi = dict(lexi[0], **lexi[1], **lexi[2])
    pinyin = Pinyin()
    def _get_phrase_vector(item: str) -> np.ndarray:

        index = ex_vocab.get(pinyin.get_pinyin(item, '')).index
        ex_vector = ex_vectors[index]

        vector = []
        count = 0
        for s in item:
            vector.append(vectors[vocab.get(s).index] * vocab.get(s).count)
            count += vocab.get(s).count
        vector = np.array(vector)
        vector = np.sum(vector, axis=0) / count

        return vector

    words = ''
    for item in lexi.keys():
        words = words + item

    uword = []
    for word in words:
        if word not in uword:
            uword.append(word)

    word_dict = dict()
    for word in uword:
        re = []
        rev = []
        for (item, count) in lexi.items():
            if item.find(word) != -1:
                re.append({
                    'item': item,
                    'count': count
                })
                rev.append(_get_phrase_vector(item))
        word_dict.update({word: {
            'r_words': re,
            'r_words_v': rev,
            'count': len(re),
        }})

    return word_dict


def add_lexi_infomation(lexi_dict: dict, vocab: dict, vectors: np.ndarray) -> np.ndarray:

    def get_weight_index(key: str, value: dict, type='B') -> np.ndarray:

        r_words = np.array(value.get('r_words'))
        r_words_v = np.array(value.get('r_words_v'))

        if type == 'B':
            v_index = []
            for index, item in enumerate(r_words):
                if key == item['item'][0]:
                    v_index.append(index)
            v_count = np.array([item['count'] for item in r_words[v_index]]).reshape((-1, 1))
            vector = np.sum(v_count * r_words_v[v_index], axis=0)
        elif type == 'I':
            v_index = []
            for index, item in enumerate(r_words):
                if len(item['item']) > 2 and key != item['item'][0] and key != item['item'][-1]:
                    v_index.append(index)
            v_count = np.array([item['count'] for item in r_words[v_index]]).reshape((-1, 1))
            vector = np.sum(v_count * r_words_v[v_index], axis=0)
        elif type == 'E':
            v_index = []
            for index, item in enumerate(r_words):
                if key == item['item'][-1]:
                    v_index.append(index)
            v_count = np.array([item['count'] for item in r_words[v_index]]).reshape((-1, 1))
            vector = np.sum(v_count * r_words_v[v_index], axis=0)

        return vector

    _vectors = np.zeros((vectors.shape[0], 2 * vectors.shape[1]))
    _vectors[:, :vectors.shape[1]] = vectors
    for key, value in lexi_dict.items():
        # combine word information
        total_count = 0
        for item in value.get('r_words'):
            total_count += item['count']
        b_v = get_weight_index(key, value, type='B')
        I_v = get_weight_index(key, value, type='I')
        E_v = get_weight_index(key, value, type='E')
        word_weight_vector = b_v + I_v + E_v
        word_weight_vector = word_weight_vector / total_count

        id = vocab.get(key)
        _vectors[id, vectors.shape[1]:] = word_weight_vector
    return _vectors.astype(np.float32)


class PostProcess:

    def __init__(self, train: np.ndarray, vocab: dict, vectors: np.ndarray, pre_train: bool = True):

        if pre_train:
            self.vocab = vocab
            self.vectors = vectors
            self.l_crops, self.l_diseases, self.l_medicines = build_lexi(train[:, 1:])
            self.l_crops, self.l_diseases, self.l_medicines = list(self.l_crops.keys()), list(self.l_diseases.keys()), list(self.l_medicines.keys())
            self.lv_corps = self._build_phrase_vectors(self.l_crops)
            self.lv_diseases = self._build_phrase_vectors(self.l_diseases)
            self.lv_medicines = self._build_phrase_vectors(self.l_medicines)

    def _build_phrase_vectors(self, lexi: list) -> np.ndarray:

        lexi_vector = []
        for item in lexi:
            lexi_vector.append(self._get_phrase_vector(item))

        return np.array(lexi_vector)

    def _delete_symbol(self, entitys: list) -> list:

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

    def _get_phrase_vector(self, item: str):

        vectors = []
        for s in item:
            vectors.append(self.vectors[self.vocab.get(s)])
        vectors = np.array(vectors)

        return np.mean(vectors, axis=0)

    def _calculate_similarity(self, entity: str, vectors: np.ndarray) -> Tuple:

        vector = self._get_phrase_vector(entity)
        similar_matrix = (vectors.dot(vector[:, np.newaxis]) / (np.linalg.norm(vectors, axis=1, keepdims=True) *
                                                               np.linalg.norm(vector))).reshape(-1)
        top = np.argmax(similar_matrix)
        return similar_matrix[top], top

    def _correct_error_item(self, entitys: list, lexi: list, phrase_vectors: np.ndarray) -> list:

        for in1, item in enumerate(entitys):
            for in2, entity in enumerate(item):
                if entity not in lexi:
                    score, top = self._calculate_similarity(entity, phrase_vectors)
                    t = lexi[top]
                    if score > 0.99:
                        entitys[in1][in2] = lexi[top]
        return entitys

    def _correct_error(self, entitys: list, index: int) -> list:
        '''
        :param entitys: object
        :param index: 0: crops, 1: diseases, 2: medicines
        :return: entitys
        '''
        if index == 0:
            entitys = self._correct_error_item(entitys, self.l_crops, self.lv_corps)
        elif index == 1:
            entitys = self._correct_error_item(entitys, self.l_diseases, self.lv_diseases)
        elif index == 2:
            entitys = self._correct_error_item(entitys, self.l_medicines, self.lv_medicines)
        else:
            pass
        return entitys

    def forward(self, entitys: list, index: int) -> list:
        entitys = self._delete_symbol(entitys)
        # entitys = self._correct_error(entitys, index)
        return entitys

    def __call__(self, *args, **kwargs):

        return self.forward(*args, **kwargs)


# ================================= main ================================= #
if __name__ == '__main__':

    train = pd.read_csv('{}/train/train.csv'.format(DATA_DIR), index_col=0).values
    test = pd.read_csv('{}/test/test.csv'.format(DATA_DIR)).values
    test_seqs = test[:, 1].tolist()

    VECTOR_SIZE = 128
    pre_train = True
    USING_CRF = False
    train_seqs, train_char_labels, word_datas = get_process_data(train)
    TRAIN_LENGTH = len(train_seqs) - 200
    vocab, token, vectors, r_vocab = build_corpus(test_seqs, train_seqs, pre_train, VECTOR_SIZE)

    if pre_train:
        lexi = build_lexi(train[:, 1:])
        word_vector, word_vocab = build_word(word_datas)
        pinyin_vector, pinyin_vocab = build_word_pinyin(word_datas)
        lexi_dict = change_lexi2dict(lexi, pinyin_vocab, pinyin_vector, r_vocab, vectors.copy())
        vectors = add_lexi_infomation(lexi_dict, vocab, vectors.copy())
        VECTOR_SIZE = vectors.shape[1]
    # token = token[:len(train_seqs)]
    # train_data = token[:TRAIN_LENGTH - 100]
    # train_label = train_char_labels[:TRAIN_LENGTH - 100]
    # valid_data = token[TRAIN_LENGTH - 100:TRAIN_LENGTH + 100]
    # valid_label = train_char_labels[TRAIN_LENGTH - 100:TRAIN_LENGTH + 100]
    # test_data = token[TRAIN_LENGTH + 100:]
    # test_label = train_char_labels[TRAIN_LENGTH + 100:]
    #
    # filename = './datas/train.txt'
    # with open(filename, 'w') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
    #     for in1, item in enumerate(train_data):
    #         for in2, entity in enumerate(item):
    #             f.write("{} {}\n".format(entity, train_label[in1][in2]))
    #         f.write("\n")
    #
    # filename = './datas/dev.txt'
    # with open(filename, 'w') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
    #     for in1, item in enumerate(valid_data):
    #         for in2, entity in enumerate(item):
    #             f.write("{} {}\n".format(entity, valid_label[in1][in2]))
    #         f.write("\n")
    #
    # filename = './datas/test.txt'
    # with open(filename, 'w') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
    #     for in1, item in enumerate(test_data):
    #         for in2, entity in enumerate(item):
    #             f.write("{} {}\n".format(entity, test_label[in1][in2]))
    #         f.write("\n")

    train_char_ids, train_char_labels, test_char_ids, category_dict = seq2id(token, vocab, train_char_labels, len(train_seqs), USING_CRF)

    test_l_ids = train_char_ids[TRAIN_LENGTH:]
    test_l_labels = train_char_labels[TRAIN_LENGTH:]
    train_char_labels = train_char_labels[:TRAIN_LENGTH]
    train_char_ids = train_char_ids[:TRAIN_LENGTH]

    post_process = PostProcess(train, vocab, vectors.copy(), pre_train)

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    TRAIN = 'TRAIN'
    VALID = 'VALID'
    TEST = 'TEST'
    HIDDEN_DIM = 320
    OUTPUT_SIZE = len(category_dict)
    BATCH_SIZE = 64
    EM_DIM = VECTOR_SIZE
    LR = 1e-2
    INPUT_SIZE = len(vocab)
    NUM_LAYERS = 1
    EPOCH = 50
    pre_model = torch.from_numpy(vectors) if pre_train else None

    # init model
    if USING_CRF:
        model = BiLSTM_CRF(INPUT_SIZE, HIDDEN_DIM, EM_DIM, OUTPUT_SIZE, NUM_LAYERS,
                           pre_model=pre_model).to(
            DEVICE)
    else:
        model = BiLSTM(INPUT_SIZE, HIDDEN_DIM, EM_DIM, OUTPUT_SIZE, NUM_LAYERS,
                           pre_model=pre_model).to(
            DEVICE)
    # init optimization
    optim = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.5, 0.99))
    lr_s = StepLR(optim, step_size=20, gamma=0.1)
    # init criterion
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    # get data iterator
    print(1)
    fold_index = 0  # using 10-fold cross validation 5
    (train_dataloader, valid_dataloader, test_dataloader) = get_data_loader(
        root='',
        data_type_name=[TRAIN, VALID, TEST],
        batch_size=[BATCH_SIZE, 5000, 5000],
        fold_index=fold_index,
        train_length=TRAIN_LENGTH,
        train_char_labels=train_char_labels,
        train_char_ids=train_char_ids,
        test_char_ids=test_char_ids,
    )

    valid_seqs = np.array(train_seqs)[get_10_fold_index(fold_index, TRAIN_LENGTH)[1]]

    print('begin training ......')
    best_model = None
    best_score = 0
    for epoch in range(EPOCH):
        # training
        model.train()
        train_loss = 0
        # if (epoch + 1) % 2 == 0:
        #     fold_index = (fold_index + 1) % 10
        #     (train_dataloader, valid_dataloader) = get_data_loader(
        #         root='',
        #         data_type_name=[TRAIN, VALID, TEST],
        #         batch_size=[BATCH_SIZE, 5000, 5000],
        #         fold_index=fold_index,
        #         train_length=TRAIN_LENGTH,
        #         train_char_labels=train_char_labels,
        #         train_char_ids=train_char_ids,
        #         test_char_ids=test_char_ids,
        #         test=False,
        #     )
        #     valid_seqs = np.array(train_seqs)[get_10_fold_index(fold_index, TRAIN_LENGTH)[1]]
        for index, batch in enumerate(train_dataloader):
            data, lengths = tensorized(batch[:, 0], vocab)
            label, _ = tensorized(batch[:, 1], category_dict)
            data, label = data.to(DEVICE), label.to(DEVICE)

            model.zero_grad()
            optim.zero_grad()

            output = model(data, lengths)
            loss = model.caculate_loss(output, label, category_dict, criterion)

            loss.backward()
            optim.step()

            train_loss += loss.item()
        lr_s.step()
        # validation
        model.eval()
        with torch.no_grad():
            batch = next(iter(valid_dataloader))
            data, lengths = tensorized(batch[:, 0], vocab)
            label, _ = tensorized(batch[:, 1], category_dict)
            data, label = data.to(DEVICE), label.to(DEVICE)
            output = model(data, lengths)
            pred = model.get_word_id(output, category_dict, lengths)
            valid_loss = model.caculate_loss(output, label, category_dict, criterion)
            p, r, f, f_acc = caculate_f_acc(*drop_entity(pred, valid_seqs, category_dict, post_process=post_process),
                                     true_labels=np.array(train)[get_10_fold_index(fold_index, TRAIN_LENGTH)[1], 1:])

            if best_score < f_acc:
                best_score = f_acc
                best_model = model.state_dict()
        print('epoch: {}, train_loss: {:.4f}, valid_loss: {:.4f}, f_acc: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1_score: {:.3f}'.
              format(epoch, train_loss / len(train_dataloader), valid_loss, f_acc, p, r, f))

    print('best_score: {:.3f}'.format(best_score))
    torch.save(best_model, './model.pkl')
    model.load_state_dict(torch.load('./model.pkl', map_location=DEVICE))
    # testing
    model.eval()
    with torch.no_grad():
        data, lengths = tensorized(test_l_ids, vocab)
        label, _ = tensorized(test_l_labels, category_dict)
        data, label = data.to(DEVICE), label.to(DEVICE)
        output = model(data, lengths)
        pred = model.get_word_id(output, category_dict, lengths)
        p, r, f, f_acc = caculate_f_acc(*drop_entity(pred, train_seqs[TRAIN_LENGTH:], category_dict, post_process=post_process),
                                 true_labels=np.array(train)[TRAIN_LENGTH:, 1:])
        print('tset: f_acc: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1_score: {:.3f}'.format(f_acc, p, r, f))

    # work
    model.eval()
    with torch.no_grad():
        batch = next(iter(test_dataloader))
        data, lengths = tensorized(batch, vocab)
        data = data.to(DEVICE)

        output = model(data, lengths)
        pred = model.get_word_id(output, category_dict, lengths)

        crops, diseases, medicines = drop_entity(pred, test_seqs, category_dict, post_process=post_process)
        pd.DataFrame([test[:, 0], crops, diseases, medicines]).T. \
            to_csv('./result.csv', header=['id', 'n_crop', 'n_disease', 'n_medicine'], index=False)
