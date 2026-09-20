#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 2024/10/14 13:26
# @Author: hb925
# @File: akt.py
'''
@inproceedings{ghoshContextAwareAttentiveKnowledge2020,
  type = {10.1145/3394486.3403282},
  title = {Context-{{Aware Attentive Knowledge Tracing}}},
  booktitle = {Proceedings of the 26th {{ACM SIGKDD International Conference}} on {{Knowledge Discovery}} \&amp; {{Data Mining}}},
  author = {Ghosh, Aritra and Heffernan, Neil and Lan, Andrew S.},
  date = {2020},
  pages = {2330--2339},
  publisher = {Association for Computing Machinery},
  location = {Virtual Event, CA, USA},
  url = {https://doi.org/10.1145/3394486.3403282},
  isbn = {978-1-4503-7998-4},
  keywords = {/code,/impl,/note,AKT,knowledge tracing personalized learning monotonic attention item response theory},
}
'''
import math

import torch
from torch.nn import Embedding, Dropout, Linear, ReLU
import torch.nn.functional as F

from torch_kt.models.akt import Architecture
from torch_kt.training import BaseKt


class AKTNP(BaseKt):
    def __init__(self, skill_num, emb_size=128, attention_blocks=4, dropout=0.1, d_ff=256, final_fc_dim=256,
                 att_heads=8,
                 l2=1e-5):
        super().__init__("AKT-NP")
        self.skill_num = skill_num
        self.dropout = dropout
        self.l2 = l2
        self.embed_dim = emb_size
        self.final_fc_dim = final_fc_dim

        self.q_embed_layer = Embedding(self.skill_num, self.embed_dim)
        self.qa_embed_layer = Embedding(2 * self.skill_num, self.embed_dim)  # interaction emb

        # Architecture Object. It contains stack of attention block
        self.arch_model = Architecture(n_blocks=attention_blocks, n_heads=att_heads, dropout=dropout,
                                       d_model=self.embed_dim,
                                       d_ff=d_ff, kq_same=1)

        self.out = torch.nn.Sequential(
            Linear(self.embed_dim + self.embed_dim, final_fc_dim), ReLU(), Dropout(self.dropout),
            Linear(final_fc_dim, 256), ReLU(
            ), Dropout(self.dropout),
            Linear(256, 1)
        )

    def forward(self, x, mask=None, training=None, **kwargs):
        qa, q = x
        q_embed = self.q_embed_layer(q)
        qa_embed = self.qa_embed_layer(qa)

        q_embed_data = q_embed
        qa_embed_data = qa_embed

        # BS.seqlen,emb_size
        # Pass to the decoder
        # output shape BS,seqlen,emb_size or emb_size//2
        d_output = self.arch_model(q_embed_data, qa_embed_data)

        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q).squeeze(-1)
        m = torch.nn.Sigmoid()
        preds = m(output)
        return preds[:, 1:, None]

    @property
    def inputs_specs(self):
        return ("skill_response", "skill"), ("correct",)

    def data_map(self, data):
        (skill_response, skill), y = data
        mask = torch.ge(y, 0).type(torch.int8)
        skill_response = (skill_response * mask).type(torch.long)
        skill = (skill * mask).type(torch.long)
        y = y.unsqueeze(-1).type(torch.float)
        return (skill_response, skill), y[:, 1:], mask.unsqueeze(-1).type(torch.bool)[:, 1:], None
