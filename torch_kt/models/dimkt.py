#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/1/15 0015 上午 10:37
# @Author  : hb
# @File    : dimkt.py
import torch
import torch.nn as nn
from torch import cat
from torch.nn import Embedding, Dropout, Linear

from torch_kt.training import BaseKt


class DIMKT(BaseKt):
    def __init__(self, skill_num, problem_num, emb_size=128, difficult_levels=100, dropout=0.1):
        super().__init__("DIMKT")
        self.num_q = problem_num
        self.num_c = skill_num
        self.emb_size = emb_size
        self.difficult_levels = difficult_levels
        self.dropout = Dropout(dropout)

        self.knowledge = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, self.emb_size)), requires_grad=True)

        self.q_emb = Embedding(self.num_q, self.emb_size)
        self.c_emb = Embedding(self.num_c, self.emb_size)
        self.sd_emb = Embedding(self.difficult_levels, self.emb_size)
        self.qd_emb = Embedding(self.difficult_levels, self.emb_size)
        self.a_emb = Embedding(2, self.emb_size)

        self.linear_1 = Linear(4 * self.emb_size, self.emb_size)
        self.linear_2 = Linear(1 * self.emb_size, self.emb_size)
        self.linear_3 = Linear(1 * self.emb_size, self.emb_size)
        self.linear_4 = Linear(2 * self.emb_size, self.emb_size)
        self.linear_5 = Linear(2 * self.emb_size, self.emb_size)
        self.linear_6 = Linear(4 * self.emb_size, self.emb_size)

    def forward(self, x, mask=None, training=None, **kwargs):
        q, c, sd, qd, a = x
        qshft = q[:, 1:]
        cshft = c[:, 1:]
        sdshft = sd[:, 1:]
        qdshft = qd[:, 1:]
        B, T = q.shape
        device=q.device
        q_emb = self.q_emb(q)
        c_emb = self.c_emb(c)
        sd_emb = self.sd_emb(sd)
        qd_emb = self.qd_emb(qd)
        a_emb = self.a_emb(a)

        target_q = self.q_emb(qshft)
        target_c = self.c_emb(cshft)
        target_sd = self.sd_emb(sdshft)
        target_qd = self.qd_emb(qdshft)

        input_data = cat((q_emb, c_emb, sd_emb, qd_emb), -1)
        input_data = self.linear_1(input_data)
        target_data = cat((target_q, target_c, target_sd, target_qd), -1)
        target_data = self.linear_1(target_data)

        k = self.knowledge.repeat(B, 1).to(q.device)

        h_list = list()
        # h.append(unsqueeze(k, dim=1))
        # h.append(unsqueeze(k, dim=1))

        seqlen = T
        for i in range(T - 1):
            sd_1 = sd_emb[:,i,:]
            a_1 = a_emb[:,i,:]
            qd_1 = qd_emb[:,i,:]
            input_data_1 = input_data[:,i,:]

            qq = input_data_1 - k

            gates_SDF = self.linear_2(qq)
            gates_SDF = torch.sigmoid(gates_SDF)
            SDFt = self.linear_3(qq)
            SDFt = torch.tanh(SDFt)
            SDFt = self.dropout(SDFt)

            SDFt = gates_SDF * SDFt

            x = cat((SDFt, a_1), -1)
            gates_PKA = self.linear_4(x)
            gates_PKA = torch.sigmoid(gates_PKA)

            PKAt = self.linear_5(x)
            PKAt = torch.tanh(PKAt)

            PKAt = gates_PKA * PKAt

            ins = cat((k, a_1, sd_1, qd_1), -1)
            gates_KSU = self.linear_6(ins)
            gates_KSU = torch.sigmoid(gates_KSU)

            k = gates_KSU * k + (1 - gates_KSU) * PKAt

            h_i = k
            # print(k.size())
            # print(h_i.size())

            h_list.append(h_i)
        # print('out')
        output = torch.stack(h_list, dim=1)
        # print(output.size())
        # logits = torch.sum(target_data * output[:,:-2,:], dim=-1)
        logits = torch.sum(target_data * output, dim=-1,keepdim=True)

        y = torch.sigmoid(logits)
        return y

    @property
    def inputs_specs(self):
        return ("skill", "problem", f"problem_diff_level_{self.difficult_levels}", f"skill_diff_level_{self.difficult_levels}"), ("correct",)

    def data_map(self, data):
        (skill, problem, k_dif_level, p_dif_level), y = data
        mask = torch.ge(y, 0).to(torch.int8)
        problem = (problem * mask).type(torch.long)
        skill = (skill * mask).type(torch.long)
        k_dif_level = (k_dif_level * mask).type(torch.long)
        p_dif_level = (p_dif_level * mask).type(torch.long)
        r = (y * mask).type(torch.long)
        y = y.unsqueeze(-1).type(torch.float)
        return (problem, skill, k_dif_level, p_dif_level, r), y[:, 1:], mask.unsqueeze(-1).type(torch.bool)[:, 1:], None
