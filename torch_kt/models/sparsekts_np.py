#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 2026/2/4 0:16
# @Author: hb925
# @File: sparsekts_np.py
import torch
from torch import nn

from torch_kt.models.sparsekts import Architecture
from torch_kt.training import BaseKt
'''
@inproceedings{huangRobustKnowledgeTracing2023,
  title = {Towards Robust Knowledge Tracing Models via K-Sparse Attention},
  booktitle = {Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  author = {Huang, Shuyan and Liu, Zitao and Zhao, Xiangyu and Luo, Weiqi and Weng, Jian},
  date = {2023},
  series = {Sigir '23},
  pages = {2441--2445},
  publisher = {Association for Computing Machinery},
  location = {Taipei, Taiwan and New York, NY, USA},
  doi = {10.1145/3539618.3592073},
  isbn = {978-1-4503-9408-6},
  pagetotal = {5}
}
'''


class SparseKTSNP(BaseKt):
    def __init__(
            self,
            skill_num,
            max_len,
            emb_size=128,
            attention_blocks=4,
            dropout=0.1,
            d_ff=256,
            final_fc_dim=256,
            att_heads=8,
            separate_qa=False,
            l2=1e-5,
            sparse_ratio=0.8
    ):
        super().__init__("SparseKTS-NP")
        """
        Input:
            emb_size: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            att_heads: number of heads in multi-headed attention
            d_ff : dimension for fully conntected net inside the basic block
            kq_same: if key query same, kq_same=1, else = 0
        """
        self.n_question = skill_num
        self.dropout = dropout
        self.n_pid = -1
        self.l2 = l2
        self.separate_qa = separate_qa
        self.sparse_ratio = sparse_ratio

        embed_l = emb_size
        if self.n_pid > 0:
            # else:
            #     self.difficult_param = nn.Embedding(self.n_pid, embed_l)  # 题目难度
            self.difficult_param = nn.Embedding(self.n_pid, 1)  # 题目难度
            self.q_embed_diff = nn.Embedding(self.n_question,
                                             embed_l)  # question emb, 总结了包含当前question（concept）的problems（questions）的变化
            self.qa_embed_diff = nn.Embedding(2 * self.n_question, embed_l)  # interaction emb, 同上

        self.q_embed = nn.Embedding(self.n_question, embed_l)
        if self.separate_qa:
            self.qa_embed = nn.Embedding(2 * self.n_question, embed_l)  # interaction emb
        else:  # false default
            self.qa_embed = nn.Embedding(2, embed_l)

        # Architecture Object. It contains stack of attention block
        self.model = Architecture(
            n_blocks=attention_blocks,
            n_heads=att_heads,
            dropout=dropout,
            d_model=emb_size,
            d_ff=d_ff,
            kq_same=True,
            seq_len=max_len,
        )

        self.out = nn.Sequential(
            nn.Linear(emb_size + embed_l, final_fc_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, final_fc_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim // 2, 1),
        )
        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_pid + 1 and self.n_pid > 0:
                torch.nn.init.constant_(p, 0.0)

    def base_emb(self, q_data, target):
        q_embed_data = self.q_embed(q_data)  # BS, seqlen,  emb_size# c_ct
        if self.separate_qa:
            qa_data = q_data + self.n_question * target
            qa_embed_data = self.qa_embed(qa_data)
        else:
            # BS, seqlen, emb_size # c_ct+ g_rt =e_(ct,rt)
            qa_embed_data = self.qa_embed(target) + q_embed_data
        return q_embed_data, qa_embed_data

    def forward(self, x, mask=None, training=None, **kwargs):
        q_data, target = x
        pid_data = None

        q_embed_data, qa_embed_data = self.base_emb(q_data, target)

        if self.n_pid > 0:
            q_embed_diff_data = self.q_embed_diff(
                q_data
            )  # d_ct 总结了包含当前question（concept）的problems（questions）的变化
            pid_embed_data = self.difficult_param(pid_data)  # uq 当前problem的难度
            q_embed_data = (
                    q_embed_data + pid_embed_data * q_embed_diff_data
            )  # uq *d_ct + c_ct # question encoder
            # qa_embed_diff_data = self.qa_embed_diff(
            #     target
            # )  # f_(ct,rt) or #h_rt (qt, rt)差异向量
            # qa_embed_data = qa_embed_data + pid_embed_data * (
            #         qa_embed_diff_data + q_embed_diff_data
            # )
        d_output, attn_weights = self.model(
            q_embed_data,
            qa_embed_data,
            sparse_ratio=self.sparse_ratio,
        )

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
