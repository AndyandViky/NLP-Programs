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
try:
    import random
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F

    from typing import Tuple
    from torch import Tensor
    from torchtext.datasets import Multi30k
    from torchtext.data import Field, BucketIterator

    from utils.config import DEVICE
    from utils.utils import caculate_accuracy

except ImportError as e:
    print(e)
    raise ImportError


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

    def __init__(self, input_dim: int, emb_dim: int, enc_hid_dim: int, dec_hid_dim: int, dropout: float, verbose: int = 0):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(input_size=emb_dim, hidden_size=enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.verbose = verbose

    def forward(self, x: Tensor) -> Tuple[Tensor]:

        embedded = self.dropout(self.embedding(x))
        out_put, hidden = self.rnn(embedded)

        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        return out_put, hidden


class Decoder(nn.Module):

    def __init__(self, dec_input_dim: int, emb_dim: int, enc_hid_dim: int, dec_hid_dim: int,
                 dropout: int, attention: nn.Module, verbose: int = 0):
        super(Decoder, self).__init__()

        self.dec_input_dim = dec_input_dim
        self.attention = attention

        self.embedding = nn.Embedding(dec_input_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.out = nn.Linear(self.attention.attn_in + emb_dim, dec_input_dim)
        self.dropout = nn.Dropout(dropout)
        self.verbose = verbose

    def _weighted_encoder_rep(self, decoder_hidden: Tensor, encoder_output: Tensor) -> Tensor:

        a = self.attention(decoder_hidden, encoder_output)

        a = a.unsqueeze(1)

        encoder_outputs = encoder_output.permute(1, 0, 2)

        weighted_encoder_rep = torch.bmm(a, encoder_outputs)

        weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)

        return weighted_encoder_rep

    def forward(self, input: Tensor, decoder_hidden: Tensor, encoder_output: Tensor) -> Tuple[Tensor]:

        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        weighted_encoder_rep = self._weighted_encoder_rep(decoder_hidden, encoder_output)

        rnn_input = torch.cat((embedded, weighted_encoder_rep), dim=2)
        output, hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))

        output = self.out(torch.cat((
            output,
            weighted_encoder_rep,
            embedded
        ), dim=2).squeeze(0))

        return output, hidden.squeeze(0)


class Attention(nn.Module):

    def __init__(self, enc_hid_dim: int, dec_hid_dim: int, attn_dim: int, verbose: int = 0):
        super(Attention, self).__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn_in = (enc_hid_dim * 2) + dec_hid_dim
        self.attn = nn.Linear(self.attn_in, attn_dim)
        self.verbose = verbose

    def forward(self, decoder_hidden: Tensor, encoder_output: Tensor) -> Tensor:

        src_len = encoder_output.shape[0]

        repeat_dec_hid = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_output = encoder_output.permute(1, 0, 2)

        energy = torch.tanh(self.attn(torch.cat(
            (repeat_dec_hid, encoder_output), dim=2)))
        attention = torch.sum(energy, dim=2)

        return F.softmax(attention, dim=1)


class Seq2Seq(nn.Module):

    def __init__(self, encoder: nn.Module, decoder: nn.Module, verbose: int = 0):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.verbose = verbose

    def forward(self, src: Tensor, trg: Tensor, teacher_forcint_ratio: float = 0.5) -> Tensor:

        max_len = trg.shape[0]
        encoder_output, hidden = self.encoder(src)
        outputs = torch.zeros((max_len, trg.shape[1], self.decoder.dec_input_dim))

        output = trg[0, :]
        for i in range(max_len):
            output, hidden = self.decoder(output, hidden, encoder_output)
            outputs[i] = output
            teacher_force = random.random() < teacher_forcint_ratio
            top1 = output.max(1)[1]
            output = trg[i] if teacher_force else top1

        return outputs


# ======================= training ======================= #
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 32
DEC_EMB_DIM = 32
ENC_HID_DIM = 64
DEC_HID_DIM = 64
ATTN_DIM = 8
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)

attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)

dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec).to(DEVICE)


def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


model.apply(init_weights)

optimizer = optim.Adam(model.parameters())


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')

PAD_INDEX = TRG.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=PAD_INDEX).to(DEVICE)

for epoch in range(100):

    # training
    model.train()
    total_loss = 0
    for index, batch in enumerate(train_iterator):

        src = batch.src.to(DEVICE)
        trg = batch.trg.to(DEVICE)

        model.zero_grad()
        optimizer.zero_grad()

        output = model(src, trg)

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        print(loss.item())
        total_loss += loss.item()

    # evaluation using accuracy
    model.eval()
    with torch.no_grad():
        e_total_loss = 0
        e_total_acc = 0
        for index, batch in enumerate(valid_iterator):

            src = batch.src.to(DEVICE)
            trg = batch.trg.to(DEVICE)

            output = model(src, trg, 0)
            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            pred = torch.argmax(output, dim=1).view((-1)).detach().cpu()
            target = trg.detach().cpu()
            e_total_acc += caculate_accuracy(pred, target)

            loss = criterion(output, trg)
            e_total_loss += loss.item()

    print('train_loss: {}, valid_loss: {}, acc: {}'.format(total_loss / len(train_iterator),
                                                           e_total_loss / len(valid_iterator),
                                                           e_total_acc / len(valid_iterator)))

# testing
model.eval()
with torch.no_grad():

    test_data = next(iter(test_iterator))
    t_src, t_trg = test_data.src, test_data.trg

    encoder_output, hidden = model.encoder(t_src)
    output = t_trg[0]
    words = []
    while True:
        output, hidden = model.decoder(output, hidden, encoder_output)
        topv, topi = output.data.topk(1)
        if topi == 1:
            words.append('<EOS>')
        else:
            pass
        output = topi.squeeze()







