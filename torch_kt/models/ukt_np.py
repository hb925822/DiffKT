#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/4/22 10:32
# @Author  : hb
# @File    : ukt_np.py
from enum import IntEnum

import torch
import torch.nn as nn

from torch_kt.models.ukt import Architecture
from torch_kt.training import BaseKt


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


class UKTNP(BaseKt):
    """
    Uncertainty-aware Knowledge Tracing (UKT) model.

    UKT tracks students' knowledge states using stochastic embeddings (mean and covariance)
    to represent uncertainty in learning, with a Wasserstein-based attention mechanism to
    capture knowledge state transitions.
    """

    def __init__(self, skill_num, max_len, emb_size=128, attention_blocks=4, dropout=0.1, d_ff=256,
                 kq_same=True, final_fc_dim=256, final_fc_dim2=256, attn_heads=8, separate_qa=False,
                 use_uncertainty_aug=True, l2=1e-5, atten_type='w2'):
        super().__init__("UKT-NP")
        self.n_question = skill_num
        self.max_len = max_len
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_pid = -1
        self.l2 = l2
        self.separate_qa = separate_qa
        self.use_uncertainty_aug = use_uncertainty_aug
        self.atten_type = atten_type
        embed_l = emb_size
        self.embed_l = emb_size
        if self.n_pid > 0:
            # if emb_type.find("scalar") != -1:
            self.difficult_param = nn.Embedding(self.n_pid, 1)  # question difficulty
            # else:
            #     self.difficult_param = nn.Embedding(self.n_pid + 1, embed_l)

            self.q_embed_diff = nn.Embedding(self.n_question,
                                             embed_l)  # question emb, summarizes the changes of problems (questions) including the current question (concept)
            self.qa_embed_diff = nn.Embedding(2 * self.n_question, embed_l)  # interaction emb,

        # n_question+1 ,d_model
        self.mean_q_embed = nn.Embedding(self.n_question, embed_l)  # mean embedding
        self.cov_q_embed = nn.Embedding(self.n_question, embed_l)  # covariance embedding
        # interaction embedding
        if self.separate_qa:
            self.mean_qa_embed = nn.Embedding(2 * self.n_question, embed_l)
            self.cov_qa_embed = nn.Embedding(2 * self.n_question, embed_l)

        else:  # false default
            self.mean_qa_embed = nn.Embedding(2, embed_l)
            self.cov_qa_embed = nn.Embedding(2, embed_l)

        # Architecture Object. It contains stack of attention block
        self.model = Architecture(n_blocks=attention_blocks, n_heads=attn_heads, dropout=dropout,
                                  d_model=embed_l, d_ff=d_ff, kq_same=self.kq_same, seq_len=self.max_len)

        self.out = nn.Sequential(
            nn.Linear(embed_l + embed_l + embed_l + embed_l,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, final_fc_dim2), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim2, 1)
        )
        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_pid + 1 and self.n_pid > 0:
                torch.nn.init.constant_(p, 0.)

    def base_emb(self, q_data, target):
        q_mean_embed_data = self.mean_q_embed(q_data)  # mean embeddings for questions/KCs.
        q_cov_embed_data = self.cov_q_embed(q_data)  # covariance embeddings for questions/KCs.

        if self.separate_qa:
            qa_data = q_data + self.n_question * target
            qa_mean_embed_data = self.mean_qa_embed(qa_data)  # mean embeddings for interactions.
            qa_cov_embed_data = self.cov_qa_embed(qa_data)  # covariance embeddings for interactions.

        else:
            qa_mean_embed_data = self.mean_qa_embed(target) + q_mean_embed_data
            qa_cov_embed_data = self.cov_qa_embed(target) + q_cov_embed_data

        return q_mean_embed_data, q_cov_embed_data, qa_mean_embed_data, qa_cov_embed_data

    def forward(self, x, mask=None, training=None, **kwargs):
        q, r = x
        pid_data = None
        q_data = q
        target = r
        # Generate stochastic embeddings for questions and responses # Equation (1)
        q_mean_embed_data, q_cov_embed_data, qa_mean_embed_data, qa_cov_embed_data = self.base_emb(q_data, target)

        if self.n_pid > 0:  # have problem id
            q_embed_diff_data = self.q_embed_diff(q_data)  # d_ct 总结了包含当前question（concept）的problems（questions）的变化
            pid_embed_data = self.difficult_param(pid_data)  # uq 当前problem的难度
            q_mean_embed_data = q_mean_embed_data + pid_embed_data * \
                                q_embed_diff_data
            q_cov_embed_data = q_cov_embed_data + pid_embed_data * \
                               q_embed_diff_data

            # else:
            #     # here
            #     q_embed_diff_data = self.q_embed_diff(q_data)
            #     pid_embed_data = self.difficult_param(pid_data)  # uq 当前problem的难度
            #
            #     q_mean_embed_data = q_mean_embed_data + pid_embed_data * \
            #                         q_embed_diff_data  # uq *d_ct + c_ct # question encoder
            #     q_cov_embed_data = q_cov_embed_data + pid_embed_data * \
            #                        q_embed_diff_data
            #
            #     qa_embed_diff_data = self.qa_embed_diff(target)  # f_(ct,rt) or #h_rt (qt, rt)差异向量
            #
            #     qa_mean_embed_data = qa_mean_embed_data + pid_embed_data * \
            #                          (qa_embed_diff_data + q_embed_diff_data)
            #     qa_cov_embed_data = qa_cov_embed_data + pid_embed_data * \
            #                         (qa_embed_diff_data + q_embed_diff_data)

        # BS.seqlen,d_model

        y2, y3 = 0, 0

        mean_d_output, cov_d_output = self.model(q_mean_embed_data, q_cov_embed_data, qa_mean_embed_data,
                                                 qa_cov_embed_data, self.atten_type)
        # Calculate CL loss if in training mode
        # Calculate uncertainty measure
        activation = nn.ELU()
        temp = torch.mean(torch.mean(activation(cov_d_output) + 1, dim=-1), -1)
        # Concatenate features for final prediction
        concat_q = torch.cat([mean_d_output, cov_d_output, q_mean_embed_data, q_cov_embed_data], dim=-1)
        # else:
        #     concat_q = torch.cat([mean_d_output, mean_d_output, q_cov_embed_data, q_cov_embed_data], dim=-1)
        output = self.out(concat_q).squeeze(-1)
        m = nn.Sigmoid()
        preds = m(output)

        return preds[:, 1:].unsqueeze(dim=-1)

    @property
    def inputs_specs(self):
        return ("problem", "skill"), "correct"

    def data_map(self, data):
        (problem, skill), y = data
        mask = torch.ge(y, 0).type(torch.int8)
        skill = (skill * mask).type(torch.long)
        problem = (problem * mask).type(torch.long)
        y = y.unsqueeze(-1).type(torch.float)
        r = (y.squeeze(-1) * mask).type(torch.long)
        return (skill, r), y[:, 1:], mask.unsqueeze(-1).type(torch.bool), None
