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
from Bi_LSTM_NER import get_data_loader, caculate_f_acc, drop_entity, get_10_fold_index
from typing import Tuple
from torch import Tensor
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
            item[0] = re.sub(r'[0-9]+', ' 0 ', item[0])
            item[0] = re.sub(r'[a-zA-Z]+', ' s ', item[0])

            if item[0] == '':
                dele_index.append(ind)
        datas = np.delete(datas, dele_index, axis=0)

        def extra_info(data: np.ndarray) -> Tuple:
            char_labels = []
            for (word, label) in data:
                word = word.replace(' ', '')
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
# ================================= pre-processing data ================================= #


# ================================= model ================================= #
class BertModel(nn.Module):

    def __init__(self):
        super(BertModel, self).__init__()

        self.model = bert.from_pretrained('./datas/bert-base-chinese/')

    def forward(self, x, attention_mask=None) -> Tensor:

        return self.model(x, attention_mask=attention_mask, output_all_encoded_layers=False)[0]


class Classifier(nn.Module):
    def __init__(self, category_dict: dict, criterion):
        super(Classifier, self).__init__()

        n_cluster = len(category_dict)
        self.criterion = criterion
        self.category_dict = category_dict
        self.model = nn.Sequential(
            nn.Linear(768, n_cluster)
            # nn.Softmax()
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

    def _correct_unk(self, front, later):

        pred = 11
        cat = self.category_dict
        b_c = cat.get('B_n_crop')
        i_c = cat.get('I_n_crop')
        e_c = cat.get('E_n_crop')
        b_d = cat.get('B_n_disease')
        i_d = cat.get('I_n_disease')
        e_d = cat.get('E_n_disease')
        b_m = cat.get('B_n_medicine')
        i_m = cat.get('I_n_medicine')
        e_m = cat.get('E_n_medicine')
        o = cat.get('O')
        c_s = [b_c, i_c, e_c]
        d_s = [b_d, i_d, e_d]
        m_s = [b_m, i_m, e_m]
        # three situation, begin, inside, end
        if front == o and later == o:
            pass
        elif front is None or front == o:
            if later == i_c or later == e_c: pred = b_c
            elif later == i_d or later == e_d: pred = b_d
            elif later == i_m or later == e_m: pred = b_m
        elif later is None or later == o:
            if front == i_c or front == b_c:
                pred = e_c
            elif front == i_d or front == b_d:
                pred = e_d
            elif front == i_m or front == b_m:
                pred = e_m
        else:
            if front in c_s and later in c_s:
                pred = i_c
            elif front in d_s and later in d_s:
                pred = i_d
            elif front in m_s and later in m_s:
                pred = i_m
            else:
                if front == i_c or front == b_c:
                    pred = e_c
                elif front == i_d or front == b_d:
                    pred = e_d
                elif front == i_m or front == b_m:
                    pred = e_m
                else:
                    if later == i_c or later == e_c:
                        pred = b_c
                    elif later == i_d or later == e_d:
                        pred = b_d
                    elif later == i_m or later == e_m:
                        pred = b_m

        return pred

    def correct_unk(self, pred):

        UNK = self.category_dict.get('[UNK]')
        x, y = torch.where(pred == UNK)
        front, later = None, None
        for i, j in zip(x, y):
            if j == 0:
                pass
            elif j == len(pred[i]) - 1:
                pass
            else:
                front = pred[i, j - 1]
                later = pred[i, j + 1]
            pred[i, j] = self._correct_unk(front, later)
        return pred

    def get_word_id(self, output: Tensor) -> Tensor:
        output = torch.softmax(output, 2)
        pred = torch.argmax(output, 2)
        # pred = self.correct_unk(pred)
        return pred
# ================================= model ================================= #


def label2id(train_labels: list, tokenized_text: list) -> Tuple:

    # char to id
    def get_char_id(sequence: list, vocab: dict, index: int) -> list:
        ids = []
        for ind, char in enumerate(sequence):
            # if tokenized_text[index][ind] == '[UNK]':
            #     ids.append(vocab.get('[UNK]'))
            # else:
            ids.append(vocab.get(char))
        return ids

    category = np.unique(np.array(sum(train_labels, [])))
    category_dict = dict(
        [(c, index) for index, c in enumerate(category)]
    )
    # category_dict.update({
    #     '[UNK]': len(category_dict)
    # })

    train_labels = np.array([get_char_id(labels, category_dict, index) for index, labels in enumerate(train_labels)])

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
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    USING_CRF = False
    ENHANCE_DATA = True
    Parallel = True
    TEST_LENGTH = 0
    BATCH_SIZE = 8
    VALID_BATCH_SIZE = 20
    fold_index = 8

    train1 = pd.read_csv('{}/train/train.csv'.format(DATA_DIR), index_col=0).values
    train = pd.read_csv('{}/train/train1.csv'.format(DATA_DIR), index_col=0).values
    train = np.vstack((train1, train))

    # pre-process training data.
    _, train_seqs, _, train_char_labels = get_process_data(train, ENHANCE_DATA)
    if ENHANCE_DATA: train = np.repeat(train, 2, axis=0)
    TRAIN_LENGTH = len(train_seqs) - TEST_LENGTH

    # load pre training format tokenizer.
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_basic_tokenize=True)
    vocab = tokenizer.vocab

    # for get the true token, we add space within the 0.
    tokenized_text = [tokenizer.tokenize(i) for i in train_seqs]
    # recover training data.
    train_seqs = [item.replace(' ', '') for item in train_seqs]

    # get token id and label id.
    tokenized_text = [['[CLS]'] + item + ['[SEP]'] for item in tokenized_text]
    train_char_labels = [['[PAD]'] + item + ['[PAD]'] for item in train_char_labels]
    train_ids = np.array([tokenizer.convert_tokens_to_ids(item) for item in tokenized_text])
    train_label_ids, category_dict = label2id(train_char_labels, tokenized_text)

    # split data to train valid and test.
    test_ids = train_ids[TRAIN_LENGTH:]
    test_labels = train_label_ids[TRAIN_LENGTH:]
    train_labels = train_label_ids[:TRAIN_LENGTH]
    train_ids = train_ids[:TRAIN_LENGTH]
    valid_seqs = np.array(train_seqs)[get_10_fold_index(fold_index, TRAIN_LENGTH)[1]]

    # define model and loss function.
    xe_loss = nn.CrossEntropyLoss()
    xe_loss.to(DEVICE)
    model = BertModel().to(DEVICE)
    classifier = Classifier(category_dict, xe_loss).to(DEVICE)
    if Parallel:
        model = nn.DataParallel(model)
        classifier = nn.DataParallel(classifier)
    training = False
    if training:
        optim = BertAdam(model.parameters(), lr=1e-5)
        c_optim = Adam(classifier.parameters(), lr=1e-3)

        (train_dataloader, valid_dataloader, _) = get_data_loader(
            root='',
            data_type_name=['TRAIN', 'VALID', 'TEST'],
            batch_size=[BATCH_SIZE, VALID_BATCH_SIZE, 5000],
            fold_index=fold_index,
            train_length=TRAIN_LENGTH,
            train_char_labels=train_labels,
            train_char_ids=train_ids,
            test_char_ids=test_ids,
        )
        # training
        best_model = None
        best_cls = None
        best_score = 0
        for epoch in range(5):
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
                if Parallel: loss = loss.mean()
                loss.backward()
                optim.step()
                c_optim.step()

                train_loss += loss.item()

            # valid
            model.eval()
            with torch.no_grad():
                total_valid_loss = 0
                crops = []
                diseases = []
                medicines = []
                for index, batch in enumerate(valid_dataloader):
                    data, lengths = tensorized(batch[:, 0], vocab)
                    labels, _ = tensorized(batch[:, 1], category_dict)
                    data, labels = data.to(DEVICE), labels.to(DEVICE)

                    output = model(data, lengths)
                    pred, valid_loss = classifier(output, labels)
                    if Parallel:
                        total_valid_loss += valid_loss.mean().item()
                    else:
                        total_valid_loss += valid_loss.item()
                    pred = pred[:, 1:].data.cpu().numpy()
                    iteration = len(valid_dataloader)
                    if index == iteration - 1:
                        valid = valid_seqs[index*VALID_BATCH_SIZE:]
                    else:
                        valid = valid_seqs[index*VALID_BATCH_SIZE: (index + 1)*VALID_BATCH_SIZE]
                    crop, disease, medicine = drop_entity(pred, valid, category_dict)
                    crops, diseases, medicines = crops + crop, diseases + disease, medicines + medicine
                p, r, f, f_acc = caculate_f_acc(crops, diseases, medicines,
                                                true_labels=np.array(train)[
                                                            get_10_fold_index(fold_index, TRAIN_LENGTH)[1],
                                                            1:])

                if best_score <= f_acc:
                    best_score = f_acc
                    best_model = model.state_dict()
                    best_cls = classifier.state_dict()
            print(
                'epoch: {}, train_loss: {:.4f}, valid_loss: {:.4f}, f_acc: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1_score: {:.3f}'.
                    format(epoch, train_loss / len(train_dataloader), total_valid_loss / len(valid_dataloader), f_acc,
                           p, r, f))

        print('best_score: {:.3f}'.format(best_score))
        torch.save(best_model, './bert_model.pkl')
        torch.save(best_cls, './classifier.pkl')

        # # testing
        # model.eval()
        # classifier.eval()
        # with torch.no_grad():
        #     data, lengths = tensorized(test_ids, vocab)
        #     labels, _ = tensorized(test_labels, category_dict)
        #     data, labels = data.to(DEVICE), labels.to(DEVICE)
        #
        #     output = model(data, lengths)
        #     pred, _ = classifier(output)
        #     pred = pred[:, 1:].data.cpu().numpy()
        #     p, r, f, f_acc = caculate_f_acc(*drop_entity(pred, train_seqs[TRAIN_LENGTH:], category_dict),
        #                                     true_labels=np.array(train)[TRAIN_LENGTH:, 1:])
        #     # print(confusion_matrix(label.view(-1).data.cpu().numpy(), pred.reshape(-1)))
        #     # print(classification_report(label.view(-1).data.cpu().numpy(), pred.reshape(-1), target_names=list(category_dict.keys())[:11]))
        #     print('tset: f_acc: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1_score: {:.3f}'.format(f_acc, p, r, f))

    else:
        # work
        model.load_state_dict(torch.load('./bert_model.pkl'))
        classifier.load_state_dict(torch.load('./classifier.pkl'))
        model.eval()
        classifier.eval()
        with torch.no_grad():
            # import scipy.io as scio
            # vocabs = scio.loadmat('./vocab.mat')['data'].tolist()
            # tokenized_text = [['[CLS]'] + tokenizer.tokenize(i) + ['[SEP]'] for i in vocabs]
            # work_ids = np.array([tokenizer.convert_tokens_to_ids(item) for item in tokenized_text])
            # data, _ = tensorized(work_ids, vocab)
            # unk = torch.where(data == vocab.get('[UNK]'))[0].data.cpu().numpy()
            # data = data.to(DEVICE)
            # output = model(data)[:, 1, :].squeeze().data.cpu().numpy()
            # t = np.zeros((1, 768))
            # output[unk] = t
            # output = np.vstack((t, t, output))
            # scio.savemat('./bert.mat', {'vectors': output})

            test = pd.read_csv('{}/test/test1.csv'.format(DATA_DIR)).values
            test_seqs = test[:, 1]
            test_seqs = np.array([re.sub(r'[a-zA-Z0-9]+', ' 0 ', sequence) for sequence in test_seqs])
            work_batch = [tokenizer.tokenize(i) for i in test_seqs]
            test_seqs = [item.replace(' ', '') for item in test_seqs]
            wrok_text = [['[CLS]'] + item + ['[SEP]'] for item in work_batch]
            work_ids = np.array([tokenizer.convert_tokens_to_ids(item) for item in wrok_text])

            batch_size = 500
            data, lengths = tensorized(work_ids, vocab)
            iteration = data.size(0) // batch_size + 1
            data = data.to(DEVICE)

            crops = []
            diseases = []
            medicines = []
            for i in range(iteration):
                if i == iteration - 1:
                    begin = i*batch_size
                    end = data.size(0)
                else:
                    begin = i * batch_size
                    end = (i+1)*batch_size
                output = model(data[begin:end], lengths[begin:end])
                pred, _ = classifier(output)
                pred = pred[:, 1:].data.cpu().numpy()
                crop, disease, medicine = drop_entity(pred, test_seqs[begin:end], category_dict)
                crops, diseases, medicines = crops + crop, diseases + disease, medicines + medicine
            pd.DataFrame([test[:, 0], crops, diseases, medicines]).T. \
                to_csv('./result.csv', header=['id', 'n_crop', 'n_disease', 'n_medicine'], index=False)


if __name__ == '__main__':
    main()

