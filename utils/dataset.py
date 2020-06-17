# -*- coding: utf-8 -*-
"""
@Author: andy
@Contact: andy_viky@163.com
@Github: https://github.com/AndyandViky
@Csdn: https://blog.csdn.net/AndyViky
@File: dataset.py
@Time: 2020/6/11 下午2:22
@Desc: dataset.py
"""
try:
    import numpy as np
    import torch

    from torchtext import data
    from torchtext.vocab import Vectors
    from torch.utils.data import Dataset, DataLoader

    from utils.config import DATASETS_DIR

except ImportError as e:
    print(e)
    raise ImportError


class TranslationData(Dataset):

    def __init__(self, root, data_name, transform, train=True):
        super(TranslationData, self).__init__()

        self.root = root
        self.data_name = data_name
        self.transform = transform
        self.train = train
        self.input_data, self.d_input_data, self.target_data, self.num_encoder_tokens, self.num_decoder_tokens \
            = self.get_data()

    def __getitem__(self, item):

        input = self.transform(self.input_data[item])
        d_input = self.transform(self.d_input_data[item])
        target = self.transform(self.target_data[item])

        return input, d_input, target

    def __len__(self):
        return len(self.input_data)

    def get_data(self, data_name='cmn'):

        data = list(open('{}/{}.txt'.format(DATASETS_DIR, data_name), encoding='utf-8'))
        e_data = []
        c_data = []
        input_characters = set()
        target_characters = set()
        for line in data[: min(1000, len(data) - 1)]:
            input_text, target_text, _ = line.split('\t')
            target_text = '\t' + target_text + '\n'
            e_data.append(input_text)
            c_data.append(target_text)

            for char in input_text:
                if char not in input_characters:
                    input_characters.add(char)
            for char in target_text:
                if char not in target_characters:
                    target_characters.add(char)

        input_characters = sorted(list(input_characters))
        target_characters = sorted(list(target_characters))
        num_encoder_tokens = len(input_characters)
        num_decoder_tokens = len(target_characters)
        max_seq_encoder_len = max([len(i) for i in e_data])
        max_seq_decoder_len = max([len(i) for i in c_data])

        input_token_index = dict(
            [(char, i) for i, char in enumerate(input_characters)]
        )
        target_token_index = dict(
            [(char, i) for i, char in enumerate(target_characters)]
        )

        encoder_input_data = np.zeros(
            (len(c_data), max_seq_encoder_len, num_encoder_tokens),
            dtype='float32')
        decoder_input_data = np.zeros(
            (len(c_data), max_seq_decoder_len, num_decoder_tokens),
            dtype='float32')
        decoder_target_data = np.zeros(
            (len(c_data), max_seq_decoder_len, num_decoder_tokens),
            dtype='float32')

        for i, (input_text, target_text) in enumerate(zip(e_data, c_data)):
            for t, char in enumerate(input_text):
                encoder_input_data[i, t, input_token_index[char]] = 1.
            encoder_input_data[i, t + 1:, input_token_index[' ']] = 1.
            for t, char in enumerate(target_text):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                decoder_input_data[i, t, target_token_index[char]] = 1.
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    decoder_target_data[i, t - 1, target_token_index[char]] = 1.
            decoder_input_data[i, t + 1:, target_token_index[' ']] = 1.
            decoder_target_data[i, t:, target_token_index[' ']] = 1.
        return encoder_input_data, decoder_input_data, decoder_target_data, num_encoder_tokens, num_decoder_tokens


def get_dataloader(root, data_name, batch_size):

    dataset = TranslationData(root=root, data_name=data_name, transform=lambda data: torch.tensor(data))
    dataloader = DataLoader(
        dataset, shuffle=True, batch_size=batch_size
    )
    return dataloader, dataset.num_encoder_tokens, dataset.num_decoder_tokens
