#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 2024/8/14 11:26
# @Author: hb925
# @File: wandb_dkt.py

import torch
from torch.nn import Embedding, LSTM, Dropout, Linear
import torch.nn.functional as F
from torch_kt.training import BaseKt


class DKT(BaseKt):
    def __init__(self, skill_num, emb_size=128, hidden_units=128, dropout=0.1):
        super().__init__("DKT")
        self.num_c = skill_num
        self.emb_size = emb_size
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.interaction_emb = Embedding(self.num_c * 2, self.emb_size)
        self.lstm_layer = LSTM(self.emb_size, self.hidden_units, batch_first=True)
        self.dropout_layer = Dropout(self.dropout)
        self.out_layer = Linear(self.hidden_units, self.num_c)

    def forward(self, x, mask=None, training=None, **kwargs):
        qa, q = x
        qa_emb = self.interaction_emb(qa)
        q_onehot = F.one_hot(q, num_classes=self.num_c)
        h, _ = self.lstm_layer(qa_emb)
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        y = self.dropout_layer(y)
        y = torch.sigmoid(y)
        y = q_onehot * y
        return y.sum(dim=-1, keepdim=True)

    @property
    def inputs_specs(self):
        return ("skill_response", "skill"), ("correct",)

    def data_map(self, data):
        (skill_response, skill), y = data
        mask = torch.ge(y, 0).to(torch.int8)
        skill_response = (skill_response * mask).type(torch.long)
        skill = (skill * mask).type(torch.long)
        y = y.unsqueeze(-1).type(torch.float)
        return (skill_response[:, :-1], skill[:, 1:]), y[:, 1:], mask.unsqueeze(-1).type(torch.bool)[:, 1:], None
