# -*- coding: utf-8 -*-
"""
@Author: andy
@Contact: andy_viky@163.com
@Github: https://github.com/AndyandViky
@Csdn: https://blog.csdn.net/AndyViky
@File: translation_v2.py
@Time: 2020/6/24 下午7:59
@Desc: translation_v2.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from typing import Tuple
from torch import Tensor
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from utils.config import DEVICE


# ======================= prepare data ======================= #
SRC = Field(tokenize='spacy', tokenizer_language='de_core_news_sm', init_token='<sos>', eos_token='<eos>', lower=True)
TRG = Field(tokenize='spacy', tokenizer_language='en_core_web_sm', init_token='<sos>', eos_token='<eos>', lower=True)

train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))

SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

BATCH_SIZE = 128

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=DEVICE,
)


# ======================= model ======================= #
class Encoder(nn.Module):

    def __init__(self, input_dim: int, emb_dim: int, enc_hid_dim: int, dec_hid_dim: int, dropout: float, verbose: int):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(input_dim=emb_dim, hidden_size=enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.verbose = verbose

    def forward(self, x: Tensor) -> Tuple[Tensor]:

        embedded = self.dropout(self.embedding(x))
        out_put, hidden = self.rnn(embedded)

        hidden = torch.tanh(self.fc(torch.cat(hidden[-2, :, :], hidden[-1, :, :], dim=1)))

        return out_put, hidden


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        pass


class Attention(nn.Module):

    def __init__(self, enc_hid_dim: int, dec_hid_dim: int, attn_dim: int):
        super(Attention, self).__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn_in = (enc_hid_dim * 2) + dec_hid_dim

        self.attn = nn.Linear(self.attn_in, attn_dim)

    def forward(self, decoder_hidden: Tensor, encoder_output: Tensor) -> Tensor:

        src_len = encoder_output.shape[0]

        repeat_dec_hid = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_output = encoder_output.permute(1, 0, 2)

        energy = torch.tanh(self.attn(torch.cat(
            (repeat_dec_hid, encoder_output), dim=2)))
        attention = torch.sum(energy, dim=2)

        return F.softmax(attention, dim=1)










