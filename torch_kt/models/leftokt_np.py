#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/9/25 21:40
# @Author  : hb
# @File    : leftokt_np.py
from enum import IntEnum

import torch
import torch.nn as nn

from torch_kt.models.leftokt import Architecture
from torch_kt.training import BaseKt


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


class LeftoKTNP(BaseKt):
    def __init__(self, skill_num, max_len, emb_size=128, attention_blocks=4, dropout=0.1,
                 d_ff=256, kq_same=True, final_fc_dim=256, attn_heads=8, separate_qa=False, l2=1e-5, **kwargs):
        if len(kwargs) > 0:
            print(f"unused params for model:{kwargs}")
        super().__init__("LeftoKT-NP")
        """
        Input:
            d_model: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            attn_heads: number of heads in multi-headed attention
            d_ff : dimension for fully conntected net inside the basic block
            kq_same: if key query same, kq_same=1, else = 0
        """
        self.n_question = skill_num
        self.n_pid = -1
        self.dropout = dropout
        self.kq_same = kq_same
        self.l2 = l2
        self.separate_qa = separate_qa
        embed_l = d_model = emb_size

        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(self.n_pid, 1)  # 题目难度
            self.q_embed_diff = nn.Embedding(self.n_question,
                                             embed_l)  # question emb, 总结了包含当前question（concept）的problems（questions）的变化
            self.qa_embed_diff = nn.Embedding(2 * self.n_question, embed_l)  # interaction emb, 同上

        # n_question+1 ,d_model
        self.q_embed = nn.Embedding(self.n_question, embed_l)
        if self.separate_qa:
            self.qa_embed = nn.Embedding(2 * self.n_question, embed_l)  # interaction emb
        else:  # false default
            self.qa_embed = nn.Embedding(2, embed_l)

        # Architecture Object. It contains stack of attention block
        self.model = Architecture(n_blocks=attention_blocks, n_heads=attn_heads, dropout=dropout,
                                  d_model=d_model, d_ff=d_ff, kq_same=self.kq_same)

        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )
        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.dim() > 0 and p.size(0) == self.n_pid + 1 and self.n_pid > 0:
                # if p.size(0) == self.n_pid+1 and self.n_pid > 0:
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

    def forward(self, x, mask=None, training=None, **kwargs):
        q, p, r = x
        target = r
        q_data = q
        pid_data = None
        # Batch First
        q_embed_data, qa_embed_data = self.base_emb(q_data, target)

        pid_embed_data = None
        if self.n_pid > 0:  # have problem id
            q_embed_diff_data = self.q_embed_diff(q_data)  # d_ct 总结了包含当前question（concept）的problems（questions）的变化
            pid_embed_data = self.difficult_param(pid_data)  # uq 当前problem的难度
            q_embed_data = q_embed_data + pid_embed_data * \
                           q_embed_diff_data  # uq *d_ct + c_ct # question encoder

            qa_embed_diff_data = self.qa_embed_diff(
                target)  # f_(ct,rt) or #h_rt (qt, rt)差异向量
            if self.separate_qa:
                qa_embed_data = qa_embed_data + pid_embed_data * \
                                qa_embed_diff_data  # uq* f_(ct,rt) + e_(ct,rt)
            else:
                qa_embed_data = qa_embed_data + pid_embed_data * \
                                (
                                        qa_embed_diff_data + q_embed_diff_data)  # + uq *(h_rt+d_ct) # （q-response emb diff + question emb diff）
            c_reg_loss = (pid_embed_data ** 2.).sum() * self.l2  # rasch部分loss
        else:
            c_reg_loss = torch.tensor(0.0, dtype=torch.float, device=q_data.device)

        # BS.seqlen,d_model
        # Pass to the decoder
        # output shape BS,seqlen,d_model or d_model//2
        d_output = self.model(q_embed_data, qa_embed_data, pid_embed_data)

        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q).squeeze(-1)
        m = nn.Sigmoid()
        preds = m(output)
        preds = preds[:, 1:].unsqueeze(dim=-1)
        return preds

    @property
    def inputs_specs(self):
        return ("problem", "skill"), "correct"

    def data_map(self, data):
        (problem, skill), y = data
        mask = torch.ge(y, 0).type(torch.bool)
        problem = (problem * mask).type(torch.long)
        skill = (skill * mask).type(torch.long)
        y = y.unsqueeze(-1).type(torch.float)
        r = (y.squeeze(-1) * mask).type(torch.long)
        return (skill, problem, r), y[:, 1:], mask.unsqueeze(-1).type(torch.bool), None
