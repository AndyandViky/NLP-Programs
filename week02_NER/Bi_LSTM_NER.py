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
import scipy.io as scio
# import synonyms

from torch.nn import LSTM
from torchcrf import CRF
from torch import Tensor
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from xpinyin import Pinyin
from gensim.models import Word2Vec
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from sklearn.metrics import confusion_matrix, classification_report
# !python -m elmoformanylangs.biLM train --train_path ./token.raw --config_path ./cnn_50_100_512_4096_sample.json --model ./QA.model --optimizer adam --lr 0.001 --lr_decay 0.8 --max_epoch 10 --max_sent_len 200 --max_vocab_size 150000 --min_count 3 --gpu 0


def get_punctuation() -> str:

    z_punctuation = '[' + string.punctuation + u' a-zA-Z0-9·！，+？＂＃＄％%＆＇（）＊★＋－／：＜＝＞＠［＼］＾＿｀｛｜｝～' \
                                               u'｟｠｢｣､〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.]'
    return z_punctuation


def get_params(crf: bool = False) -> Tuple:

    # VECTOR_SIZE, pre_train, HIDDEN_DIM, BATCH_SIZE, LR, NUM_LAYERS, EPOCH, STEP_SIZE, GAMMA
    config = {
        0: (128, True, 320, 64, 1e-2, 1, 70, 10, 0.5),
        1: (128, True, 128, 64, 1e-3, 1, 30, 10, 0.9),
    }
    print(1)
    return config[int(crf)]


# ================================= pre-processing data ================================= #
def get_process_data(train: list, enhance_data: str) -> Tuple:

    entity = ['n_disease', 'n_crop', 'n_medicine']
    # split data and labels
    def split_raw_data(sequence: str, index: int) -> Tuple:
        replace_s = ['，', '%', '％', '+', '＋', '·', '：', '、', '；', '?', '—', '！', '。', '（', '）']
        for s in replace_s:
            sequence = sequence.replace(s, '{}/p '.format(s))
        raw_arr = sequence.split(' ')
        datas = np.array([i.split('/') for i in raw_arr if i != '' and len(i.split('/')) == 2])

        dele_index = []
        for ind, item in enumerate(datas):
            item[0] = item[0].replace(' ', '')
            item[0] = re.sub(r'[0-9]', '0', item[0])

            if item[0] == '':
                dele_index.append(ind)
        datas = np.delete(datas, dele_index, axis=0)

        def extra_info(data: np.ndarray) -> Tuple:
            char_labels = []
            for (word, label) in data:
                if label not in entity:
                    char_labels = char_labels + ['O' for i in range(len(word))]
                else:
                    char_labels.append('B_{}'.format(label))
                    char_labels = char_labels + ['I_{}'.format(label) for i in range(len(word) - 2)]
                    char_labels.append('E_{}'.format(label))

            seq_data = ''.join(data[:, 0])
            seq_pro = ' '.join(data[:, 1])

            return data, seq_data, seq_pro, char_labels

        if enhance_data:
            # instead_datas = datas.copy()
            # random_index = np.random.randint(0, len(datas), 10)
            # random_index = list(set(random_index))
            # for index in random_index:
            #     if datas[index][1] not in entity and datas[index][0] not in (get_punctuation() + '，。；？'):
            #         syn_word = synonyms.nearby(datas[index][0])[0]
            #         if len(syn_word) > 1: instead_datas[index][0] = syn_word[1]

            exchange_datas = datas.copy()
            for ite in range(8):
                random_index = np.random.randint(0, len(datas), 2)
                random_index = list(set(random_index))
                if len(random_index) == 2:
                    t = exchange_datas[random_index[0]].copy()
                    exchange_datas[random_index[0]] = exchange_datas[random_index[1]]
                    exchange_datas[random_index[1]] = t

            # insert_datas = datas.copy()
            # while True:
            #     random_index = np.random.randint(0, len(datas), 1)
            #     insert = insert_datas[random_index]
            #     if insert[0][0] not in (get_punctuation() + '，。；？'):
            #         break
            # insert_index = list(set(np.random.randint(0, len(datas), 3)))
            # for index in insert_index:
            #     insert_datas = np.insert(insert_datas, index, insert, axis=0)

            return extra_info(datas), extra_info(exchange_datas)
        else:
            return extra_info(datas)

    np.random.seed(0)
    post_data = np.array([split_raw_data(item, ind) for ind, item in enumerate(train[:, 0])])
    if enhance_data: post_data = post_data.reshape((-1, post_data.shape[2]))
    return post_data[:, 0], post_data[:, 1], post_data[:, 2], post_data[:, 3]


def build_corpus(test_seqs: np.ndarray, train_seqs: np.ndarray, use_word: bool = False, vector_size: int = 150
                 ) -> Tuple:
    # combine test data to build vocab
    seq_datas = train_seqs.tolist() + test_seqs.tolist()

    def char_split(sentence: str) -> list:
        s_arr = []
        for word in sentence:
            s_arr.append(word)
        return s_arr

    # calculate word2vector
    token = [char_split(i) for i in seq_datas]
    if use_word:
        token.insert(0, ['<pad>', '<unk>'])
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
        return vocab, token, [], []


def seq2id(token: list, vocab: dict, train_char_labels: list) -> Tuple:

    train_length = len(train_char_labels)
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

    train_char_labels = np.array([get_char_id(labels, category_dict) for labels in train_char_labels])

    return train_char_ids, train_char_labels, test_char_ids, category_dict
# ================================= pre-processing data ================================= #


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
        # self.train = train
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
# ================================= data ================================= #


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

    def get_word_id(self, output: Tensor) -> Tensor:

        output = torch.softmax(output, 2)
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
        self.crf = CRF(output_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: Tensor, lengths: list) -> Tensor:

        embeded = self.dropout(self.embedding(x))
        packed = pack_padded_sequence(embeded, lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.rnn(packed)
        output, hidden = pad_packed_sequence(output, batch_first=True)
        output = self.out(output)

        return output

    def caculate_loss(self, output: Tensor, target: Tensor, category_dict: dict, _) -> Tensor:

        PAD = category_dict['<pad>']
        assert PAD is not None

        mask = (target != PAD)  # [B, L]

        return - self.crf(output, target, mask)

    def get_word_id(self, output: Tensor) -> Tensor:

        """使用维特比算法进行解码"""
        output = np.array(self.crf.decode(output))
        return output
# ================================= model ================================= #


# ================================= util functions ================================= #
def drop_entity(pred: np.ndarray, test_seqs: list, category_dict: dict, post_process=None) -> Tuple:

    b_crop, i_crop, e_crop = category_dict['B_n_crop'], category_dict['I_n_crop'], category_dict['E_n_crop']

    b_disease, i_disease, e_disease = category_dict['B_n_disease'], category_dict['I_n_disease'], category_dict[
        'E_n_disease']

    b_medicine, i_medicine, e_medicine = category_dict['B_n_medicine'], category_dict['I_n_medicine'], category_dict[
        'E_n_medicine']

    def get_entity_id(seq_item: list, item: np.ndarray, b_e, i_e, e_e) -> list:
        begin = np.where(item == b_e)[0]
        result = []
        if len(begin) == 0: return result
        for index, j in enumerate(begin):
            if j >= len(seq_item): break
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
                        break
            if item[last_index] == e_e:
                result.append(seq)
        return result

    crops = [get_entity_id(test_seqs[index], item, b_crop, i_crop, e_crop) for index, item in enumerate(pred)]
    diseases = [get_entity_id(test_seqs[index], item, b_disease, i_disease, e_disease) for index, item in enumerate(pred)]
    medicines = [get_entity_id(test_seqs[index], item, b_medicine, i_medicine, e_medicine) for index, item in enumerate(pred)]

    if post_process is None:
        return crops, diseases, medicines
    else:
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

    def get_lexi(entitys: np.ndarray) -> Tuple:
        l_entitys = []
        l_entity_c = []

        for item in entitys:
            for i in eval(item):
                if i not in l_entitys:
                    l_entitys.append(i)
                    l_entity_c.append(1)
                else:
                    l_entity_c[l_entitys.index(i)] += 1

        return zip(l_entitys, l_entity_c)

    l_crops = dict(
        [(key, value) for key, value in get_lexi(crops)]
    )
    l_diseases = dict(
        [(key, value) for key, value in get_lexi(diseases)]
    )
    l_medicines = dict(
        [(key, value) for key, value in get_lexi(medicines)]
    )
    return l_crops, l_diseases, l_medicines


def build_word(word_datas: np.ndarray, size: int) -> Tuple:

    words = [item[:, 0].tolist() for item in word_datas]

    model = Word2Vec(words, size=size, window=15, min_count=0).wv
    word_vector = model.vectors
    vocab = model.vocab
    return word_vector, vocab


def build_word_pinyin(word_datas: np.ndarray) -> Tuple:

    pinyin = Pinyin()
    words = [[pinyin.get_pinyin(p, '') for p in item[:, 0].tolist()] for item in word_datas]
    model = Word2Vec(words, size=128, window=10, min_count=0).wv
    pinyin_vector = model.vectors
    pinyin_vocab = model.vocab

    return pinyin_vector, pinyin_vocab


def change_lexi2dict(lexi: Tuple, ex_vocab: dict, ex_vectors: np.ndarray, vocab: dict, vectors: np.ndarray) -> dict:

    lexi = dict(lexi[0], **lexi[1], **lexi[2])
    pinyin = Pinyin()
    def _get_phrase_vector(item: str) -> np.ndarray:

        index = ex_vocab.get(item).index
        ex_vector = ex_vectors[index]

        vector = np.array([vectors[vocab.get(s).index] for s in item])
        vector = np.mean(vector, axis=0)

        return 0.9 * vector + 0.1 * ex_vector

    words = ''.join(lexi.keys())
    uword = set(words)

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

    def get_weight_index(key: str, value: dict, type: str = 'B') -> np.ndarray:

        r_words = np.array(value.get('r_words'))
        r_words_v = np.array(value.get('r_words_v'))

        if type == 'B':
            v_index = [index for index, item in enumerate(r_words) if key == item['item'][0]]
        elif type == 'I':
            v_index = [index for index, item in enumerate(r_words) if
                       len(item['item']) > 2 and key != item['item'][0] and key != item['item'][-1]]
        elif type == 'E':
            v_index = [index for index, item in enumerate(r_words) if key == item['item'][-1]]

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

    def __init__(self, lexi: Tuple,
                 test: list,
                 vocab: dict,
                 vectors: np.ndarray,
                 char_vocab: dict,
                 char_vectors: np.ndarray,
                 pre_train: bool = True,
                 ids: list = []):

        if pre_train:
            self.vocab = vocab
            self.vectors = vectors
            self.char_vocab = char_vocab
            self.char_vectors = char_vectors
            self.l_crops, self.l_diseases, self.l_medicines = lexi
            self.l_crops, self.l_diseases, self.l_medicines = sorted(list(self.l_crops.keys()), key=lambda x: len(x), reverse=True), \
                                                              sorted(list(self.l_diseases.keys()), key=lambda x: len(x), reverse=True), \
                                                              sorted(list(self.l_medicines.keys()), key=lambda x: len(x), reverse=True)
            self.lv_corps = self._build_phrase_vectors(self.l_crops)
            self.lv_diseases = self._build_phrase_vectors(self.l_diseases)
            self.lv_medicines = self._build_phrase_vectors(self.l_medicines)

            self.lexi_result = self._get_result_from_lexi(test)
            # pd.DataFrame([ids, self.lexi_result[0], self.lexi_result[1], self.lexi_result[2]]).T. \
            #     to_csv('./l_result.csv', header=['id', 'n_crop', 'n_disease', 'n_medicine'], index=False)
            # print(1)

    def _get_result_from_lexi(self, test: list) -> Tuple:

        te_crops = []
        te_diseases = []
        te_medicines = []
        def _get_item(sequence: str, lexi_item: list) -> Tuple:
            entitys = []
            for entity in lexi_item:
                c_index = find_all(sequence, entity)
                if c_index == -1:
                    continue
                else:
                    for index in c_index:
                        entitys.append(sequence[index: index + len(entity)])
                    sequence = sequence.replace(entity, ',')
            return entitys, sequence

        for sequence in test:
            entitys, sequence = _get_item(sequence, self.l_medicines)
            if '芸苔素' in entitys:
                entitys.remove('芸苔素')
            te_medicines.append(entitys)
            entitys, sequence = _get_item(sequence, self.l_diseases)
            te_diseases.append(entitys)
            entitys, sequence = _get_item(sequence, self.l_crops)
            te_crops.append(entitys)

        return te_crops, te_diseases, te_medicines

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

    def _get_phrase_vector(self, item: str) -> np.ndarray:

        vector = []
        for i in item:
            vector.append(self.char_vectors[self.char_vocab.get(i)])
        return np.mean(np.array(vector), axis=0)
        # if item not in self.vocab.keys():
        #     vector = []
        #     for i in item:
        #         vector.append(self.char_vectors[self.char_vocab.get(i)])
        #     return np.mean(np.array(vector), axis=0)
        # else:
        #     return self.vectors[self.vocab.get(item).index]

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

    def _merge_entity(self, entitys: list, lexi_result: list):
        for index, item in enumerate(entitys):
            if len(item) < len(lexi_result[index]):
                entitys[index] = list(set(item + lexi_result[index]))
        return entitys

    def forward(self, entitys: list, index: int) -> list:
        entitys = self._delete_symbol(entitys)
        # entitys = self._correct_error(entitys, index)
        # entitys = self._merge_entity(entitys, self.lexi_result[index])
        return entitys

    def __call__(self, *args, **kwargs):

        return self.forward(*args, **kwargs)


def get_wiki_vectors(vocab: list) -> np.ndarray:

    raw = open('{}/newsblogbbs.vec'.format(DATA_DIR))
    line = raw.readline()
    vectors = np.zeros((len(vocab), 200))
    while True:
        line = raw.readline()
        if line == '':
            break
        line = line.split(' ')
        if line[0] in vocab[2:]:
            vectors[vocab.index(line[0])] = np.array(line[1:-1], dtype=np.float32)

    scio.savemat('{}/w2v_wiki.mat'.format(DATA_DIR), {'vectors': vectors})
    return vectors


def get_elmo_vector(vocab: list) -> np.ndarray:

    from elmoformanylangs import Embedder
    e = Embedder('{}/zhs.model'.format(DATA_DIR))

    vectors = np.zeros((len(vocab), 1024))
    vectors[2:] = e.sents2elmo(vocab[2:])
    scio.savemat('{}/elmo.mat'.format(DATA_DIR), {'vectors': vectors})
    return vectors


def get_bert_vector(vocab: list) -> np.ndarray:

    from bert_serving.client import BertClient
    bc = BertClient()

    vectors = np.zeros((len(vocab), 768))
    vectors[2:1066] = bc.encode(vocab[2:1066])
    vectors[1067:] = bc.encode(vocab[1067:])
    scio.savemat('{}/bert.mat'.format(DATA_DIR), {'vectors': vectors})
    return vectors


def get_wiki_bc(vocab: list) -> np.ndarray:

    raw = open('{}/w2v_wiki.bigram-char'.format(DATA_DIR))
    line = raw.readline()
    vectors = np.zeros((len(vocab), 300))
    while True:
        line = raw.readline()
        if line == '':
            break
        line = line.split(' ')
        if line[0] in vocab[2:]:
            vectors[vocab.index(line[0])] = np.array(line[1:-1], dtype=np.float32)

    scio.savemat('{}/w2v_wiki_bc.mat'.format(DATA_DIR), {'vectors': vectors})
    return vectors


def find_all(source: str, dest: str) -> list:

    length1, length2 = len(source), len(dest)
    dest_list = []
    temp_list = []
    if length1 < length2:
        return -1
    i = 0
    while i <= length1-length2:
        if source[i] == dest[0]:
            dest_list.append(i)
        i += 1
    if dest_list == []:
        return -1
    for x in dest_list:
        # print("Now x is:%d. Slice string is :%s" % (x, repr(source[x:x+length2])), end=" ")
        if source[x:x+length2] != dest:
            # print(" dest != slice")
            temp_list.append(x)
        else:
            pass
            # print(" dest == slice")
    for x in temp_list:
        dest_list.remove(x)
    return dest_list


def enhance_data(train_data: np.ndarray, lexi, test_seqs) -> np.ndarray:

    # diseases = sorted(lexi[1], key=lambda x: lexi[1].get(x))[:80]
    # diseases = ['抗病品种', '潜叶蛾', '地下虫害', '花叶病', '斑潜蝇', '蜗牛危害', '速效性', '斑潜蝇危害', '阔叶杂草', '蔓枯病', '纹裂果', '金龟子', '地下害虫', '积累中毒']
    diseases = ['蚧壳虫', '低温冻害', '烟青虫', '花叶病', '斑潜蝇', '盲蝽象', '夜蛾', '烧根', '气害灼伤', '积累中毒', '少量畸形果', '美洲斑潜蝇', '细菌性叶斑病',
                '菌核病']
    # medicines = sorted(lexi[2], key=lambda x: lexi[2].get(x))[:80]
    # medicines = ['敌百虫', '十恶霉灵', '硼钙肥', '草胺', '甲萘威', '乐果', '十烯酰吗啉', '吡丙醚', '复合肥', '十螯合钙', '敌敌畏']
    medicines = ['铜制剂', '虫酰肼', '氟啶脲', '多杀霉素', '虱螨脲', '农用链霉素', '矮壮素', '硼钙元素', '杀虫单', '硝酸钾', '除草剂残留']

    def get_relative_index(entity: list) -> list:
        insert_index = []
        for index, item in enumerate(train_data):
            has_entity = False
            for j in entity:
                if item[0].find(j) != -1:
                    has_entity = True
                    break
            if has_entity:
                insert_index.append(index)
        return insert_index

    dise_index = get_relative_index(diseases)
    medi_idnex = get_relative_index(medicines)

    insert_index = list(set(dise_index + medi_idnex))
    enhance_part = np.repeat(train_data[insert_index], 10, axis=0)

    return enhance_part
# ================================= util functions ================================= #


# ================================= main ================================= #
if __name__ == '__main__':

    DATA_DIR = './datas/QA_data'
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    TRAIN = 'TRAIN'
    VALID = 'VALID'
    TEST = 'TEST'
    USING_CRF = True
    ENHANCE_DATA = True
    TEST_LENGTH = 0
    VECTOR_SIZE, pre_train, HIDDEN_DIM, BATCH_SIZE, LR, NUM_LAYERS, EPOCH, STEP_SIZE, GAMMA = get_params(USING_CRF)

    train1 = pd.read_csv('{}/train/train.csv'.format(DATA_DIR), index_col=0).values
    train = pd.read_csv('{}/train/train1.csv'.format(DATA_DIR), index_col=0).values

    train = np.vstack((train1, train))
    test = pd.read_csv('{}/test/test1.csv'.format(DATA_DIR)).values
    test_seqs = test[:, 1]
    test_seqs = np.array([re.sub(r'[0-9]', '0', sequence) for sequence in test_seqs])

    # ======================== enhance data ========================= #
    # enhance_part = enhance_data(train[:len(train) - TEST_LENGTH], build_lexi(train[:, 1:]), test_seqs)
    # train = np.vstack((enhance_part, train))
    # ======================== enhance data ========================= #

    word_datas, train_seqs, seq_pros, train_char_labels = get_process_data(train, ENHANCE_DATA)
    # scio.savemat('train_seqs.mat', {'data': train_seqs, 'labels': train[:, 1:]})
    if ENHANCE_DATA: train = np.repeat(train, 2, axis=0)
    TRAIN_LENGTH = len(train_seqs) - TEST_LENGTH
    vocab, token, vectors, r_vocab = build_corpus(test_seqs, train_seqs, pre_train, VECTOR_SIZE)

    lexi = build_lexi(train[:, 1:])
    # scio.savemat('./lexi.mat', {'data': [list(lexi[0].keys()), list(lexi[1].keys()), list(lexi[2].keys())]})
    if pre_train:
        # wiki_vectors = get_wiki_vectors(list(vocab.keys()))
        # wiki_bc_vectors = get_elmo_vector(list(vocab.keys()))
        # wiki_bc_vectors = get_wiki_bc(list(vocab.keys()))
        # bert_vectors = get_bert_vector(list(vocab.keys()))
        elmo_vectors = scio.loadmat('{}/elmo.mat'.format(DATA_DIR))['vectors']
        wiki_vectors = scio.loadmat('{}/w2v_wiki.mat'.format(DATA_DIR))['vectors']
        wiki_bc_vectors = scio.loadmat('{}/w2v_wiki_bc.mat'.format(DATA_DIR))['vectors']
        # bert_vectors = scio.loadmat('{}/bert.mat'.format(DATA_DIR))['vectors']
        vectors = np.hstack((vectors, elmo_vectors[:, :VECTOR_SIZE], wiki_bc_vectors[:, :VECTOR_SIZE]))
        word_vector, word_vocab = build_word(word_datas, VECTOR_SIZE * 3)
        pinyin_vector, pinyin_vocab = build_word_pinyin(word_datas)
        lexi_dict = change_lexi2dict(lexi, word_vocab, word_vector, r_vocab, vectors.copy())
        vectors = add_lexi_infomation(lexi_dict, vocab, vectors.copy())
        VECTOR_SIZE = vectors.shape[1]

    fold_index = 8  # using 10-fold cross validation 0
    valid_seqs = np.array(train_seqs)[get_10_fold_index(fold_index, TRAIN_LENGTH)[1]]

    train_char_ids, train_char_labels, test_char_ids, category_dict = seq2id(token, vocab, train_char_labels)

    test_l_ids = train_char_ids[TRAIN_LENGTH:]
    test_l_labels = train_char_labels[TRAIN_LENGTH:]
    train_char_labels = train_char_labels[:TRAIN_LENGTH]
    train_char_ids = train_char_ids[:TRAIN_LENGTH]

    post_process = PostProcess(lexi, test_seqs, word_vocab, word_vector.copy(), vocab, vectors.copy(), pre_train, test[:, 0])
    pre_model = torch.from_numpy(vectors) if pre_train else None
    OUTPUT_SIZE = len(category_dict)
    INPUT_SIZE = len(vocab)
    EM_DIM = VECTOR_SIZE
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
    optim = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.5, 0.99), weight_decay=1e-5)
    lr_s = StepLR(optim, step_size=STEP_SIZE, gamma=GAMMA)
    # init criterion
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    # get data iterator
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

    print('begin training ......')
    best_model = None
    best_score = 0
    for epoch in range(EPOCH):
        # training
        model.train()
        train_loss = 0
        # if (epoch + 1) % 5 == 0:
        #     fold_index = (fold_index + 1) % 10
        #     print('current fold index is: {}'.format(fold_index))
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
            # nn.utils.clip_grad_norm_(model.parameters(), 5)
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
            pred = model.get_word_id(output)
            valid_loss = model.caculate_loss(output, label, category_dict, criterion)
            p, r, f, f_acc = caculate_f_acc(*drop_entity(pred, valid_seqs, category_dict),
                                     true_labels=np.array(train)[get_10_fold_index(fold_index, TRAIN_LENGTH)[1], 1:])

            if best_score <= f_acc:
                best_score = f_acc
                best_model = model.state_dict()
        print('epoch: {}, train_loss: {:.4f}, valid_loss: {:.4f}, f_acc: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1_score: {:.3f}'.
              format(epoch, train_loss / len(train_dataloader), valid_loss, f_acc, p, r, f))

    print('best_score: {:.3f}'.format(best_score))
    torch.save(best_model, './model.pkl')
    model.load_state_dict(torch.load('./model.pkl', map_location=DEVICE))
    # testing
    # model.eval()
    # with torch.no_grad():
    #     data, lengths = tensorized(test_l_ids, vocab)
    #     label, _ = tensorized(test_l_labels, category_dict)
    #     data, label = data.to(DEVICE), label.to(DEVICE)
    #     output = model(data, lengths)
    #     pred = model.get_word_id(output)
    #     p, r, f, f_acc = caculate_f_acc(*drop_entity(pred, train_seqs[TRAIN_LENGTH:], category_dict),
    #                              true_labels=np.array(train)[TRAIN_LENGTH:, 1:])
    #     print(confusion_matrix(label.view(-1).data.cpu().numpy(), pred.reshape(-1)))
    #     # print(classification_report(label.view(-1).data.cpu().numpy(), pred.reshape(-1), target_names=list(category_dict.keys())[:11]))
    #     print('tset: f_acc: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1_score: {:.3f}'.format(f_acc, p, r, f))

    # work
    model.eval()
    with torch.no_grad():
        batch = next(iter(test_dataloader))
        data, lengths = tensorized(batch, vocab)
        data = data.to(DEVICE)

        output = model(data, lengths)
        pred = model.get_word_id(output)

        crops, diseases, medicines = drop_entity(pred, test_seqs, category_dict, post_process=post_process)
        pd.DataFrame([test[:, 0], crops, diseases, medicines]).T. \
            to_csv('./result.csv', header=['id', 'n_crop', 'n_disease', 'n_medicine'], index=False)
