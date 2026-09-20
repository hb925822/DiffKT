#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/15 0015 下午 4:01
# @Author  : hb
# @File    : deep_irt.py
import torch
import torch.nn as nn
from torch.nn import Parameter, Embedding, Linear, Dropout
from torch.nn.init import kaiming_normal_
from torch_kt.training import BaseKt


class DeepIRT(BaseKt):
    def __init__(self, skill_num, memory_dim=256, memory_size=64, dropout=0.2):
        super().__init__("Deep-IRT")
        self.num_c = skill_num
        self.dim_s = memory_dim
        self.size_m = memory_size

        self.k_emb_layer = Embedding(self.num_c, self.dim_s)
        self.Mk = Parameter(torch.Tensor(self.size_m, self.dim_s))
        self.Mv0 = Parameter(torch.Tensor(self.size_m, self.dim_s))

        kaiming_normal_(self.Mk)
        kaiming_normal_(self.Mv0)

        self.v_emb_layer = Embedding(self.num_c * 2, self.dim_s)

        self.f_layer = Linear(self.dim_s * 2, self.dim_s)
        self.dropout_layer = Dropout(dropout)
        self.p_layer = Linear(self.dim_s, 1)

        self.diff_layer = nn.Sequential(Linear(self.dim_s, 1), nn.Tanh())
        self.ability_layer = nn.Sequential(Linear(self.dim_s, 1), nn.Tanh())

        self.e_layer = Linear(self.dim_s, self.dim_s)
        self.a_layer = Linear(self.dim_s, self.dim_s)

    def forward(self, x, mask=None, training=None, **kwargs):
        qa, q = x
        batch_size = q.shape[0]
        k = self.k_emb_layer(q)  # question embedding
        v = self.v_emb_layer(qa)  # q,a embedding
        Mvt = self.Mv0.unsqueeze(0).repeat(batch_size, 1, 1)
        Mv = [Mvt]
        w = torch.softmax(torch.matmul(k, self.Mk.T), dim=-1)
        # Write Process
        e = torch.sigmoid(self.e_layer(v))
        a = torch.tanh(self.a_layer(v))
        for et, at, wt in zip(
                e.permute(1, 0, 2), a.permute(1, 0, 2), w.permute(1, 0, 2)
        ):
            Mvt = Mvt * (1 - (wt.unsqueeze(-1) * et.unsqueeze(1))) + \
                  (wt.unsqueeze(-1) * at.unsqueeze(1))
            Mv.append(Mvt)
        Mv = torch.stack(Mv, dim=1)
        # Read Process
        f = torch.tanh(
            self.f_layer(
                torch.cat(
                    [
                        (w.unsqueeze(-1) * Mv[:, :-1]).sum(-2),
                        k
                    ],
                    dim=-1
                )
            )
        )
        stu_ability = self.ability_layer(self.dropout_layer(f))  # equ 12
        que_diff = self.diff_layer(self.dropout_layer(k))  # equ 13
        p = torch.sigmoid(3.0 * stu_ability - que_diff)  # equ 14
        return p[:, 1:, :]

    @property
    def inputs_specs(self):
        return ("skill_response", "skill"), ("correct",)

    def data_map(self, data):
        (skill_response, skill), y = data
        mask = torch.ge(y, 0).to(torch.int8)
        skill_response = (skill_response * mask).type(torch.long)
        skill = (skill * mask).type(torch.long)
        y = y.unsqueeze(-1).type(torch.float)
        return (skill_response, skill), y[:, 1:], mask.unsqueeze(-1).type(torch.bool)[:, 1:], None
