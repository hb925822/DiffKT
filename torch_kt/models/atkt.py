#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/15 0015 下午 5:26
# @Author  : hb
# @File    : atkt.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_kt.training import BaseKt
'''
@inproceedings{guoEnhancingKnowledgeTracing2021,
  type = {10.1145/3474085.3475554},
  title = {Enhancing {{Knowledge Tracing}} via {{Adversarial Training}}},
  booktitle = {Proceedings of the 29th {{ACM International Conference}} on {{Multimedia}}},
  author = {Guo, Xiaopeng and Huang, Zhijie and Gao, Jie and Shang, Mingyu and Shu, Maojing and Sun, Jun},
  year = 2021,
  pages = {367--375},
  publisher = {Association for Computing Machinery},
  address = {Virtual Event, China},
  isbn = {978-1-4503-8651-7},
  langid = {american},
  keywords = {,/impl,/note,adversarial training,knowledge hidden state attention,knowledge tracing},
 }
'''

class ATKT(BaseKt):
    def __init__(self, skill_num, embed_size=128, hidden_units=128, answer_dim=8, attention_dim=80, dropout=0.1):
        super().__init__("ATKT")
        self.skill_dim = embed_size
        self.answer_dim = answer_dim
        self.hidden_dim = hidden_units
        self.num_c = skill_num
        self.rnn = nn.LSTM(self.skill_dim + self.answer_dim, self.hidden_dim, batch_first=True)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(self.hidden_dim * 2, self.num_c)

        self.skill_emb = nn.Embedding(self.num_c, self.skill_dim)
        self.skill_emb.weight.data[-1] = 0

        self.answer_emb = nn.Embedding(2 + 1, self.answer_dim)
        self.answer_emb.weight.data[-1] = 0

        self.attention_dim = attention_dim
        self.mlp = nn.Linear(self.hidden_dim, self.attention_dim)
        self.similarity = nn.Linear(self.attention_dim, 1, bias=False)

    def attention_module(self, lstm_output):
        att_w = self.mlp(lstm_output)
        att_w = torch.tanh(att_w)
        att_w = self.similarity(att_w)
        attn_mask = torch.triu(torch.ones(lstm_output.shape[1],lstm_output.shape[1]),diagonal=1).to(dtype=torch.bool).to(lstm_output.device)
        att_w = att_w.transpose(1, 2).expand(lstm_output.shape[0], lstm_output.shape[1], lstm_output.shape[1]).clone()
        att_w = att_w.masked_fill_(attn_mask, float("-inf"))
        alphas = torch.nn.functional.softmax(att_w, dim=-1)
        attn_ouput = torch.bmm(alphas, lstm_output)
        #论文的原始实现没有对未来的att做屏蔽
        # alphas = nn.Softmax(dim=1)(att_w)
        # attn_ouput = alphas * lstm_output  # 整个seq的attn之和为1，计算前面的的时候，所有的attn都<<1，不会有问题？做的少的时候，历史作用小，做得多的时候，历史作用变大？
        attn_output_cum = torch.cumsum(attn_ouput, dim=1)
        attn_output_cum_1 = attn_output_cum - attn_ouput
        final_output = torch.cat((attn_output_cum_1, lstm_output), 2)
        return final_output

    def forward(self, x, mask=None, training=None, **kwargs):
        q_all, r = x
        q, q_next = q_all[:, :-1], q_all[:, 1:]
        q_onehot = F.one_hot(q_next, num_classes=self.num_c)
        skill_embedding = self.skill_emb(q)
        answer_embedding = self.answer_emb(r)
        skill_answer = torch.cat((skill_embedding, answer_embedding), 2)
        answer_skill = torch.cat((answer_embedding, skill_embedding), 2)

        answer = r.unsqueeze(2).expand_as(skill_answer)

        skill_answer_embedding = torch.where(answer == 1, skill_answer, answer_skill)
        out, _ = self.rnn(skill_answer_embedding)
        out = self.attention_module(out)
        fc = self.fc(self.dropout_layer(out))
        res = torch.sigmoid(fc)
        y = q_onehot * res
        y = y.sum(dim=-1, keepdim=True)
        return y

    @property
    def inputs_specs(self):
        return ("skill",), ("correct",)

    def data_map(self, data):
        skill, y = data
        mask = torch.ge(y, 0).to(torch.int8)
        skill = (skill * mask).type(torch.long)
        y = y.unsqueeze(-1).type(torch.float)
        r = (y[:, :-1] + 1).squeeze(-1).type(torch.long)
        return (skill, r), y[:, 1:], mask.unsqueeze(-1).type(torch.bool)[:, 1:], None
