#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/4/22 01:01
# @Author  : hb
# @File    : simplekt_np.py

import torch
import torch.nn as nn

from torch_kt.models.simplekt import Architecture
from torch_kt.training import BaseKt

'''
@inproceedings{liuSimpleKTSimpleToughtobeat2023,
  title = {simpleKT: A Simple but Tough-to-Beat Baseline for Knowledge Tracing},
  booktitle = {The Eleventh International Conference on Learning Representations},
  author = {Liu, Zitao and Liu, Qiongqiong and Chen, Jiahao and Huang, Shuyan and Luo, Weiqi},
  date = {2023},
  url = {https://openreview.net/forum?id=9HiGqC9C-KA}
}
'''


class SimpleKTNP(BaseKt):
    def __init__(self, skill_num, max_len, emb_size=128, attention_blocks=4, dropout=0.1, d_ff=256,
                 final_fc_dim=256,
                 att_heads=8, kq_same=True, separate_qa=False,
                 l2=1e-5, **kwargs):
        if len(kwargs) > 0:
            print(f"unused params for model:{kwargs}")
        super().__init__("SimpleKT-NP")
        self._losses = []
        self._labels = []
        self._outputs = []
        self.optimizer = None
        self.skill_max = skill_num
        self.max_len = max_len
        self.n_question = skill_num
        self.dropout = dropout
        self.kq_same = kq_same
        self.l2 = l2
        self.separate_qa = separate_qa
        d_model = embed_l = emb_size
        seq_len = max_len
        self.q_embed = nn.Embedding(self.n_question, embed_l)
        if self.separate_qa:
            self.qa_embed = nn.Embedding(2 * self.n_question, embed_l)
        else:  # false default
            self.qa_embed = nn.Embedding(2, embed_l)
        # Architecture Object. It contains stack of attention block
        self.model = Architecture(n_question=self.n_question, n_blocks=attention_blocks, n_heads=att_heads,
                                  dropout=dropout,
                                  d_model=d_model, d_feature=d_model / att_heads, d_ff=d_ff, kq_same=self.kq_same,
                                  seq_len=seq_len)

        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )

        # self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_pid + 1 and self.n_pid > 0:
                torch.nn.init.constant_(p, 0.)

    def base_emb(self, q_data, target):
        q_embed_data = self.q_embed(q_data)  # BS, seqlen,  d_model# c_ct
        if self.separate_qa:
            qa_data = q_data + self.n_question * target
            qa_embed_data = self.qa_embed(qa_data)
        else:
            # BS, seqlen, d_model # c_ct+ g_rt =e_(ct,rt)
            qa_embed_data = self.qa_embed(target) + q_embed_data
        return q_embed_data, qa_embed_data

    def get_attn_pad_mask(self, sm):
        batch_size, l = sm.size()
        pad_attn_mask = sm.data.eq(0).unsqueeze(1)
        pad_attn_mask = pad_attn_mask.expand(batch_size, l, l)
        return pad_attn_mask.repeat(self.nhead, 1, 1)

    def forward(self, x, mask=None, training=None, **kwargs):
        q, r = x
        q_data = q
        target = r

        q_embed_data, qa_embed_data = self.base_emb(q_data, target)

        d_output = self.model(q_embed_data, qa_embed_data)

        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q).squeeze(-1)
        m = nn.Sigmoid()
        preds = m(output)

        return preds[:, 1:].unsqueeze(dim=-1)

    @property
    def inputs_specs(self):
        return ("skill",), "correct"

    def data_map(self, data):
        skill, y = data
        mask = torch.ge(y, 0).type(torch.int8)
        skill = (skill * mask).type(torch.long)
        r = (y * mask).type(torch.long)
        y = y.unsqueeze(-1).type(torch.float)
        return (skill, r), y[:, 1:], mask.unsqueeze(-1).type(torch.bool), None
