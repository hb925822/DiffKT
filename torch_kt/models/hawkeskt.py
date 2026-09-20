#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/15 0015 下午 12:33
# @Author  : hb
# @File    : hawkeskt.py
import numpy as np
import torch
from torch_kt.training import BaseKt


class HawkesKT(BaseKt):
    def __init__(self, skill_num, problem_num, emb_size=128, time_log=2.0):
        super().__init__("HawkesKT")
        self.problem_num = problem_num
        self.skill_num = skill_num
        self.emb_size = emb_size
        self.time_log = time_log

        self.problem_base = torch.nn.Embedding(self.problem_num, 1)
        self.skill_base = torch.nn.Embedding(self.skill_num, 1)

        self.alpha_inter_embeddings = torch.nn.Embedding(self.skill_num * 2, self.emb_size)
        self.alpha_skill_embeddings = torch.nn.Embedding(self.skill_num, self.emb_size)
        self.beta_inter_embeddings = torch.nn.Embedding(self.skill_num * 2, self.emb_size)
        self.beta_skill_embeddings = torch.nn.Embedding(self.skill_num, self.emb_size)

    def forward(self, x, mask=None, times=None, training=None, **kwargs):
        qa, q, p = x
        alpha_src_emb = self.alpha_inter_embeddings(qa)  # [bs, seq_len, emb]
        alpha_target_emb = self.alpha_skill_embeddings(q)
        alphas = torch.matmul(alpha_src_emb, alpha_target_emb.transpose(-2, -1))  # [bs, seq_len, seq_len]
        beta_src_emb = self.beta_inter_embeddings(qa)  # [bs, seq_len, emb]
        beta_target_emb = self.beta_skill_embeddings(q)
        betas = torch.matmul(beta_src_emb, beta_target_emb.transpose(-2, -1))  # [bs, seq_len, seq_len]
        betas = torch.clamp(betas + 1, min=0, max=10)
        # use true timestamps
        seq_len = q.shape[1]
        if times:
            times = times.double() / 1000
            delta_t = (times[:, :, None] - times[:, None, :]).abs().double()
        else:
            delta_t = torch.ones(q.shape[0], seq_len, seq_len).double().to(q.device)
        delta_t = torch.log(delta_t + 1e-10) / np.log(self.time_log)

        cross_effects = alphas * torch.exp(-betas * delta_t)
        # valid_mask = torch.tril(torch.ones((1, seq_len, seq_len)).to(q.device)).to(torch.bool)
        # sum_t = cross_effects.masked_fill(valid_mask, 0).sum(-2)
        valid_mask = torch.ones((1, seq_len, seq_len)).triu(diagonal=1).to(q.device)
        sum_t = cross_effects.masked_fill(valid_mask == 0, 0).sum(-2).unsqueeze(dim=-1)

        problem_bias = self.problem_base(p)
        skill_bias = self.skill_base(q)
        h = problem_bias + skill_bias + sum_t
        prediction = torch.sigmoid(h).float()
        return prediction[:, 1:, :]

    @property
    def inputs_specs(self):
        return ("skill_response", "skill", "problem"), ("correct",)

    def data_map(self, data):
        (skill_response, skill, problem), y = data
        mask = torch.ge(y, 0).to(torch.int8)
        skill_response = (skill_response * mask).type(torch.long)
        skill = (skill * mask).type(torch.long)
        problem = (problem * mask).type(torch.long)
        y = y.unsqueeze(-1).type(torch.float)
        return (skill_response, skill, problem), y[:, 1:], mask.unsqueeze(-1).type(torch.bool)[:, 1:], None
