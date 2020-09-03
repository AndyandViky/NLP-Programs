# -*- coding: utf-8 -*-
"""
@Author: andy
@Contact: andy_viky@163.com
@Github: https://github.com/AndyandViky
@Csdn: https://blog.csdn.net/AndyViky
@File: rule_ner.py
@Time: 2020/9/2 下午11:04
@Desc: rule_ner.py
"""
import jieba
import scipy.io as scio
import pandas as pd
import numpy as np
import re

from typing import Tuple

from week02_NER.Bi_LSTM_NER import caculate_f_acc, find_all

DATA_DIR = './datas/QA_data'
lexi = scio.loadmat('./lexi.mat')['data'][0]
crop = [i.replace(' ', '') for i in lexi[0].tolist()]
diseases = [i.replace(' ', '') for i in lexi[1].tolist()]
diseases.remove('腐病')
medicines = [i.replace(' ', '') for i in lexi[2].tolist()]
medicines.remove('芸苔素')

test = pd.read_csv('{}/test/test1.csv'.format(DATA_DIR)).values
index = test[:, 0]
test_seqs = test[:, 1]
test_seqs = np.array([re.sub(r'[a-zA-Z0-9]', '0', sequence) for sequence in test_seqs])

train1 = pd.read_csv('{}/train/train.csv'.format(DATA_DIR), index_col=0).values
train = pd.read_csv('{}/train/train1.csv'.format(DATA_DIR), index_col=0).values

train = np.vstack((train1, train))
true_value = train[:, 1:][-500:]
train = scio.loadmat('./train_seqs.mat')
train_seqs = train['data'][0]
train_seqs = train_seqs[-500:]
train_seqs = np.array([re.sub(r'[a-zA-Z0-9]', '0', sequence[0]) for sequence in train_seqs])


def jieba_cut():
    jieba.load_userdict(crop + diseases + medicines)
    def _get_result_from_lexi(test: list) -> Tuple:
        te_crops = []
        te_diseases = []
        te_medicines = []

        def _get_item(sequence: list, lexi_item: list) -> Tuple:
            entitys = []
            for i in sequence:
                if i in lexi_item:
                    entitys.append(i)
            return entitys

        for sequence in test:
            result = jieba.lcut(sequence)
            entitys = _get_item(result, medicines)
            te_medicines.append(entitys)
            entitys = _get_item(result, diseases)
            te_diseases.append(entitys)
            entitys = _get_item(result, crop)
            te_crops.append(entitys)

        return te_crops, te_diseases, te_medicines

    entity_result = _get_result_from_lexi(train_seqs)
    # pd.DataFrame([index, entity_result[0], entity_result[1], entity_result[2]]).T. \
    #             to_csv('./rule_result.csv', header=['id', 'n_crop', 'n_disease', 'n_medicine'], index=False)
    p, r, f, f_acc = caculate_f_acc(entity_result[0], entity_result[1], entity_result[2], true_value)

    print(p, r, f, f_acc)
jieba_cut()


def large_match():
    jieba.load_userdict(crop + diseases + medicines)
    def _build_large_dict(entitys: list) -> dict:

        result = dict()
        for item in entitys:
            if item[:2] not in result.keys():
                result.update({
                    item[:2]: [len(item)]
                })
            else:
                if len(item) not in result.get(item[:2]): result.get(item[:2]).append(len(item))

        return result

    large_crop = _build_large_dict(crop)
    large_disease = _build_large_dict(diseases)
    large_medicine = _build_large_dict(medicines)

    te_crops = []
    te_diseases = []
    te_medicines = []
    def _judge_complete(cut_s: list, entity: str, ind: int):

        first = entity[0]
        end = entity[-1]
        if entity in cut_s:
            return True
        count = -1
        for index, item in enumerate(cut_s):
            count += len(item)
            if count >= ind:
                if first in item[1:]:
                    return False
                break
        count = 0
        for index, item in enumerate(cut_s):
            count += len(item)
            if count >= ind + len(entity):
                if end in item[:-1]:
                    return False
                break
        return True

    def _get_item(sequence: str, lexi_item: list, entitys_lexi: list) -> Tuple:
        entitys = []
        keys = lexi_item.keys()
        for entity in keys:
            c_index = find_all(sequence, entity)
            if c_index == -1:
                continue
            else:
                remove_list = []
                for index in c_index:
                    e_len = sorted(lexi_item.get(entity), reverse=True)
                    for l in e_len:
                        s = sequence[index: index + l]
                        if s in entitys_lexi and _judge_complete(jieba.lcut(sequence), s, index):
                            entitys.append(s)
                            # remove_list.append(replace)
                            break
                # remove_list = list(set(remove_list))
                # for j in remove_list:
                #     sequence = sequence.replace(j, ',')
        return entitys, sequence

    for sequence in train_seqs:
        entitys, sequence = _get_item(sequence, large_medicine, medicines)
        te_medicines.append(entitys)
        entitys, sequence = _get_item(sequence, large_disease, diseases)
        te_diseases.append(entitys)
        entitys, sequence = _get_item(sequence, large_crop, crop)
        te_crops.append(entitys)

    # pd.DataFrame([index, te_crops, te_diseases, te_medicines]).T. \
    #             to_csv('./rule_result_lm.csv', header=['id', 'n_crop', 'n_disease', 'n_medicine'], index=False)
    p, r, f, f_acc = caculate_f_acc(te_crops, te_diseases, te_medicines, true_value)

    print(p, r, f, f_acc)

large_match()
