#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 2024/8/14 13:43
# @Author: hb925
# @File: sakt.py

import torch
from torch.nn import Embedding, Dropout, Linear, MultiheadAttention, LayerNorm, Sequential, ReLU, ModuleList

from torch_kt.training import BaseKt


class AttBlock(torch.nn.Module):
    def __init__(self, emb_size, num_attn_heads, dropout) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.num_attn_heads = num_attn_heads
        self.dropout = dropout
        self.attn = MultiheadAttention(self.emb_size, self.num_attn_heads, dropout=self.dropout, batch_first=True)
        self.attn_dropout = Dropout(self.dropout)
        self.attn_layer_norm = LayerNorm(self.emb_size)

        self.FFN = Sequential(
            Linear(self.emb_size, self.emb_size),
            ReLU(),
            Dropout(self.dropout),
            Linear(self.emb_size, self.emb_size),
            # Dropout(self.dropout),
        )
        self.FFN_dropout = Dropout(self.dropout)
        self.FFN_layer_norm = LayerNorm(self.emb_size)

    def forward(self, q=None, k=None, v=None, mask=None):
        attn_mask = torch.ones(k.shape[1], v.shape[1]).to(dtype=torch.bool).to(k.device)
        attn_mask = attn_mask.triu(diagonal=1)
        # if mask is not None:
        #     mask=mask.permute(0,2,1).repeat([self.att_heads, v.shape[1],1]).to(dtype=torch.bool).logical_not()
        #     attn_mask=attn_mask.logical_or(mask)
        attn_emb, _ = self.attn(q, k, v, attn_mask=attn_mask)
        attn_emb = self.attn_dropout(attn_emb)
        attn_emb = self.attn_layer_norm(q + attn_emb)
        emb = self.FFN(attn_emb)
        emb = self.FFN_dropout(emb)
        emb = self.FFN_layer_norm(attn_emb + emb)
        return emb


class SAKT(BaseKt):
    def __init__(self, skill_num, max_len, emb_size=128, attention_blocks=4, att_heads=4, dropout=0.1):
        super().__init__("SAKT")
        self.num_c = skill_num
        self.max_len = max_len
        self.attention_blocks = attention_blocks
        self.emb_size = emb_size
        self.att_heads = att_heads
        self.dropout = dropout
        self.interaction_emb = Embedding(self.num_c * 2, self.emb_size)
        self.skill_emb = Embedding(self.num_c, self.emb_size)
        self.position_emb = Embedding(self.max_len, emb_size)
        self.blocks = ModuleList(
            [AttBlock(self.emb_size, self.att_heads, self.dropout) for _ in range(self.attention_blocks)])
        self.dropout_layer = Dropout(self.dropout)
        self.out_layer = Linear(self.emb_size, 1)
        self.register_buffer("indexs", torch.arange(self.max_len - 1).unsqueeze(0))

    def forward(self, x, mask=None, training=None, **kwargs):
        qa, q = x
        qa_emb = self.interaction_emb(qa)
        pos_emb = self.position_emb(self.get_buffer("indexs"))
        q_emb = self.skill_emb(q)
        x_emb = qa_emb + pos_emb
        for i in range(self.attention_blocks):
            x_emb = self.blocks[i](q_emb, x_emb, x_emb, mask=mask)
        h = self.dropout_layer(x_emb)
        y = self.out_layer(h)
        y = torch.sigmoid(y)
        return y

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
