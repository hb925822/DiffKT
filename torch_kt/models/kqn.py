#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/15 0015 下午 12:06
# @Author  : hb
# @File    : kqn.py
import torch
import torch.nn.functional as F
from torch.nn import Dropout, Linear, ReLU, LSTM

from torch_kt.training import BaseKt


class KQN(BaseKt):
    def __init__(self, skill_num, hidden_units=128, mlp_hidden=128, rnn_hidden=128, dropout=0.1, **kwargs):
        if len(kwargs) > 0:
            print(f"unused params for model:{kwargs}")
        super().__init__("KQN")
        self.num_c = skill_num
        self.hidden_units = hidden_units
        self.rnn_hidden = rnn_hidden
        self.mlp_hidden = mlp_hidden
        self.dropout = dropout
        self.dropout_layer = Dropout(self.dropout)
        self.rnn = LSTM(
            input_size=2 * self.num_c,
            hidden_size=self.rnn_hidden,
            num_layers=3,
            batch_first=True, dropout=self.dropout
        )
        # self.rnn = nn.GRU(
        #     input_size=2 * n_skills,
        #     hidden_units=n_rnn_hidden,
        #     num_layers=n_rnn_layers,
        #     batch_first=True
        # )
        self.linear = Linear(self.rnn_hidden, self.hidden_units)
        self.dropout_layer = Dropout(self.dropout)
        self.skill_encoder = torch.nn.Sequential(
            Linear(self.num_c, self.mlp_hidden),
            ReLU(),
            Linear(self.mlp_hidden, self.hidden_units),
            ReLU()
        )
        self.interaction_eye = torch.eye(2 * self.num_c)
        self.skill_eye = torch.eye(self.num_c)

    # def init_hidden(self, batch_size: int):
    #     weight = next(self.parameters()).data
    #     if self.rnn_type == 'lstm':
    #         return (Variable(weight.new(self.n_rnn_layers, batch_size, self.n_rnn_hidden).zero_()),
    #                 Variable(weight.new(self.n_rnn_layers, batch_size, self.n_rnn_hidden).zero_()))
    #     else:
    #         return Variable(weight.new(self.n_rnn_layers, batch_size, self.n_rnn_hidden).zero_())

    def forward(self, x, mask=None, training=None, **kwargs):
        qa, q = x
        in_data = self.interaction_eye.to(q.device)[qa]
        next_skills = self.skill_eye.to(q.device)[q]
        encoded_knowledge = self.encode_knowledge(in_data.to(q.device))  # (batch_size, max_seq_len, n_hidden)
        encoded_skills = self.encode_skills(next_skills.to(q.device))  # (batch_size, max_seq_len, n_hidden)
        encoded_knowledge = self.dropout_layer(encoded_knowledge)
        logits = torch.sum(encoded_knowledge * encoded_skills, dim=2, keepdim=True)  # (batch_size, max_seq_len)
        logits = torch.sigmoid(logits)
        return logits

    def encode_knowledge(self, in_data):
        # batch_size = in_data.size(0)
        # self.hidden = self.init_hidden(batch_size)

        rnn_output, _ = self.rnn(in_data)
        # rnn_output, _ = pad_packed_sequence(rnn_output, batch_first=True) # (batch_size, max_seq_len, n_rnn_hidden)
        encoded_knowledge = self.linear(rnn_output)  # (batch_size, max_seq_len, n_hidden)
        return encoded_knowledge

    def encode_skills(self, next_skills):
        encoded_skills = self.skill_encoder(next_skills)  # (batch_size, max_seq_len, n_hidden)
        encoded_skills = F.normalize(encoded_skills, p=2, dim=2)  # L2-normalize
        return encoded_skills

    @property
    def inputs_specs(self):
        return ("skill_response", "skill"), ("correct",)

    def data_map(self, data):
        (skill_response, skill), y = data
        mask = torch.ge(y, 0).to(torch.int8)
        skill_response = (skill_response * mask).type(torch.long)
        skill = (skill * mask).type(torch.long)
        y = y.unsqueeze(-1).type(torch.float)
        return (skill_response[:, :-1], skill[:, 1:]), y[:, 1:], mask.unsqueeze(-1).type(torch.bool)[:, 1:, :], None
