#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/16 0016 下午 1:14
# @Author  : hb
# @File    : stablekt_np.py

import torch
from torch import nn

from torch_kt.models.stablekt import Architecture
from torch_kt.training import BaseKt


class StableKTNP(BaseKt):
    def __init__(self, skill_num,
                 emb_size=128, attention_blocks=4, dropout=0.1, d_ff=256, final_fc_dim=256,
                 att_heads=8, separate_qa=False, l2=1e-5, penumbra_r=0.1, penumbra_g=0.1):
        super().__init__("StableKT-NP")
        assert emb_size // att_heads >= 1
        self.n_question = skill_num
        self.dropout = dropout
        self.n_pid = -1
        self.l2 = l2
        self.separate_qa = separate_qa
        embed_l = emb_size
        self.r = penumbra_r
        self.gamma = penumbra_g

        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(self.n_pid, 1)  # problem difficulty
            # else:
            #     self.difficult_param = nn.Embedding(self.n_pid, embed_l)  # problem difficulty
            self.q_embed_diff = nn.Embedding(self.n_question, embed_l)  # question emb
            self.qa_embed_diff = nn.Embedding(2 * self.n_question, embed_l)  # interaction emb

        # skill_num+1 ,emb_size
        self.q_embed = nn.Embedding(self.n_question, embed_l)
        if self.separate_qa:
            self.qa_embed = nn.Embedding(2 * self.n_question, embed_l)
        else:  # false default
            self.qa_embed = nn.Embedding(2, embed_l)
        # Architecture Object. It contains stack of attention block
        self.model = Architecture(n_blocks=attention_blocks, n_heads=att_heads, dropout=dropout, d_model=emb_size,
                                  d_ff=d_ff, kq_same=True, r=self.r, gamma=self.gamma)

        self.out = nn.Sequential(
            nn.Linear(emb_size + embed_l,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, final_fc_dim // 2), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim // 2, 1)
        )

        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_pid + 1 and self.n_pid > 0:
                torch.nn.init.constant_(p, 0.)

    def base_emb(self, q_data, target):
        q_embed_data = self.q_embed(q_data)  # BS, seqlen,  emb_size# c_ct
        if self.separate_qa:
            qa_data = q_data + self.n_question * target
            qa_embed_data = self.qa_embed(qa_data)
        else:
            # BS, seqlen, emb_size # c_ct+ g_rt =e_(ct,rt)
            qa_embed_data = self.qa_embed(target) + q_embed_data
        return q_embed_data, qa_embed_data

    def get_attn_pad_mask(self, sm):
        batch_size, l = sm.size()
        pad_attn_mask = sm.data.eq(0).unsqueeze(1)
        pad_attn_mask = pad_attn_mask.expand(batch_size, l, l)
        return pad_attn_mask.repeat(self.nhead, 1, 1)

    def forward(self, x, mask=None, training=None, **kwargs):
        q_data, target = x
        pid_data = None
        # Batch First
        q_embed_data, qa_embed_data = self.base_emb(q_data, target)
        if self.n_pid > 0:  # have problem id
            q_embed_diff_data = self.q_embed_diff(q_data)
            pid_embed_data = self.difficult_param(pid_data)
            q_embed_data = q_embed_data + pid_embed_data * \
                           q_embed_diff_data

            # else:
            #     q_embed_diff_data = self.q_embed_diff(q_data)
            #     pid_embed_data = self.difficult_param(pid_data)
            #     q_embed_data = q_embed_data + pid_embed_data * \
            #                    q_embed_diff_data
            #
            #     qa_embed_diff_data = self.qa_embed_diff(
            #         target)
            #     qa_embed_data = qa_embed_data + pid_embed_data * \
            #                     (qa_embed_diff_data + q_embed_diff_data)

        # BS.seqlen,emb_size
        # Pass to the decoder
        # output shape BS,seqlen,emb_size or emb_size//2
        d_output = self.model(q_embed_data, qa_embed_data)

        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q).squeeze(-1)
        preds = torch.sigmoid(output)
        preds = preds[:, 1:].unsqueeze(dim=-1)
        if training:
            return preds
        else:
            return preds

    @property
    def inputs_specs(self):
        return ("skill",), ("correct",)

    def data_map(self, data):
        skill, y = data
        mask = torch.ge(y, 0).to(torch.int8)
        skill = (skill * mask).type(torch.long)
        r = (y * mask).type(torch.long)
        y = y.unsqueeze(-1).type(torch.float)
        return (skill, r), y[:, 1:], mask.unsqueeze(-1).type(torch.bool)[:, 1:], None
