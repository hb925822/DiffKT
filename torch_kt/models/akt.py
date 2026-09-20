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
from torch_kt.training import BaseKt


# class LearnablePositionalEmbedding(torch.nn.Module):
#     def __init__(self, emb_size, max_len=512):
#         super().__init__()
#         # Compute the positional encodings once in log space.
#         pe = 0.1 * torch.randn(max_len, emb_size)
#         pe = pe.unsqueeze(0)
#         self.weight = torch.nn.Parameter(pe, requires_grad=True)
#
#     def forward(self, x):
#         return self.weight[:, :x.size(1), :]  # ( 1,seq,  Feature)
#
#
# class CosinePositionalEmbedding(torch.nn.Module):
#     def __init__(self, emb_size, max_len=512):
#         super().__init__()
#         # Compute the positional encodings once in log space.
#         pe = 0.1 * torch.randn(max_len, emb_size)
#         position = torch.arange(0, max_len).unsqueeze(1).float()
#         div_term = torch.exp(torch.arange(0, emb_size, 2).float() *
#                              -(math.log(10000.0) / emb_size))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
#         self.weight = torch.nn.Parameter(pe, requires_grad=False)
#
#     def forward(self, x):
#         return self.weight[:, :x.size(1), :]  # ( 1,seq,  Feature)


class MonotonicMultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.h = n_heads
        self.kq_same = kq_same
        self.dropout = dropout

        self.v_linear = torch.nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = torch.nn.Linear(d_model, d_model, bias=bias)
        self.dropout_layer = torch.nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = torch.nn.Linear(d_model, d_model, bias=bias)
        self.gammas = torch.nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.torch.nn.init.xavier_uniform_(self.gammas)

    def forward(self, q, k, v, mask, zero_pad, pdiff=None):

        bs = q.size(0)
        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * emb_size

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores = self.attention(q, k, v, mask,  zero_pad, pdiff)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output

    def attention(self, q, k, v, mask, zero_pad, pdiff=None):
        """
        This is called by Multi-head atention object to find the values.
        """
        device = q.device
        # d_k: 每一个头的dim
        scores = torch.matmul(q, k.transpose(-2, -1)) / \
                 math.sqrt(self.d_k)  # BS, 8, seqlen, seqlen
        bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

        x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
        x2 = x1.transpose(0, 1).contiguous()

        with torch.no_grad():
            scores_ = scores.masked_fill(mask == 0, -1e32)
            scores_ = F.softmax(scores_, dim=-1)  # BS,8,seqlen,seqlen
            scores_ = scores_ * mask.float().to(device)  # 结果和上一步一样
            distcum_scores = torch.cumsum(scores_, dim=-1)  # bs, 8, sl, sl
            disttotal_scores = torch.sum(
                scores_, dim=-1, keepdim=True)  # bs, 8, sl, 1 全1
            # print(f"distotal_scores: {disttotal_scores}")
            position_effect = torch.abs(
                x1 - x2)[None, None, :, :].type(torch.FloatTensor).to(device)  # 1, 1, seqlen, seqlen 位置差值
            # bs, 8, sl, sl positive distance
            dist_scores = torch.clamp(
                (disttotal_scores - distcum_scores) * position_effect, min=0.)  # score <0 时，设置为0
            dist_scores = dist_scores.sqrt().detach()
        m = torch.nn.Softplus()
        gamma = -1. * m(self.gammas).unsqueeze(0)  # 1,8,1,1 一个头一个gamma参数， 对应论文里的theta
        # Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e5
        if pdiff == None:
            total_effect = torch.clamp(torch.clamp(
                (dist_scores * gamma).exp(), min=1e-5), max=1e5)  # 对应论文公式1中的新增部分
        else:
            diff = pdiff.unsqueeze(1).expand(pdiff.shape[0], dist_scores.shape[1], pdiff.shape[1], pdiff.shape[2])
            diff = diff.sigmoid().exp()
            total_effect = torch.clamp(torch.clamp(
                (dist_scores * gamma * diff).exp(), min=1e-5), max=1e5)  # 对应论文公式1中的新增部分
        scores = scores * total_effect

        scores.masked_fill_(mask == 0, -1e32)
        scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
        # print(f"before zero pad scores: {scores.shape}")
        # print(zero_pad)
        if zero_pad:
            pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
            scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)  # 第一行score置0
        # print(f"after zero pad scores: {scores}")
        scores = self.dropout_layer(scores)
        output = torch.matmul(scores, v)
        # import sys
        # sys.exit()
        return output


class AktTransformerLayer(torch.nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout, kq_same):
        super().__init__()
        """
            This is a Basic Block of Transformer paper. It containts one Multi-head attention object. Followed by layer norm and postion wise feedforward net and dropout layer.
        """
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MonotonicMultiHeadAttention(
            d_model, n_heads, dropout, kq_same=kq_same)

        # Two layer norm layer and two droput layer
        self.layer_norm1 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)

        self.linear1 = torch.nn.Linear(d_model, d_ff)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(d_ff, d_model)

        self.layer_norm2 = torch.nn.LayerNorm(d_model)
        self.dropout2 = torch.nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True, pdiff=None):
        """
        Input:
            block : object of type BasicBlock(nn.Module). It contains masked_attn_head objects which is of type MultiHeadAttention(nn.Module).
            mask : 0 means, it can peek only past values. 1 means, block can peek only current and pas values
            query : Query. In transformer paper it is the input for both encoder and decoder
            key : Keys. In transformer paper it is the input for both encoder and decoder
            Values. In transformer paper it is the input for encoder and  encoded output for decoder (in masked attention part)

        Output:
            query: Input gets changed over the layer and returned.

        """

        seqlen, batch_size = query.size(1), query.size(0)
        src_mask = torch.tril(
            torch.ones((1, 1, seqlen, seqlen), device=query.device, dtype=torch.int8), diagonal=mask - 1)
        # mask == 0, zero-padding is needed.# 只能看到之前的信息，当前的信息也看不到，此时会把第一行score全置0，表示第一道题看不到历史的interaction信息，第一题attn之后，对应value全0
        # Calls block.masked_attn_head.forward() method
        query2 = self.masked_attn_head(
            query, key, values, mask=src_mask, zero_pad=True if mask==0 else False, pdiff=pdiff)

        query = query + self.dropout1((query2))  # 残差1
        query = self.layer_norm1(query)  # layer norm
        if apply_pos:
            query2 = self.linear2(self.dropout(  # FFN
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2))  # 残差
            query = self.layer_norm2(query)  # lay norm
        return query


class Architecture(torch.nn.Module):
    def __init__(self, n_blocks, d_model, d_ff, n_heads, dropout, kq_same):
        super().__init__()
        """
            n_block : number of stacked blocks in the attention
            emb_size : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = emb_size
        """
        self.d_model = d_model
        self.blocks_1 = torch.nn.ModuleList([
            AktTransformerLayer(d_model=d_model, d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
            for _ in range(n_blocks)
        ])
        self.blocks_2 = torch.nn.ModuleList([
            AktTransformerLayer(d_model=d_model, d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
            for _ in range(n_blocks * 2)
        ])

    def forward(self, q_embed_data, qa_embed_data, pid_embed_data=None):
        # target shape  bs, seqlen
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_pos_embed

        # encoder
        for block in self.blocks_1:  # encode qas, 对0～t-1时刻前的qa信息进行编码
            y = block(mask=1, query=y, key=y, values=y, pdiff=pid_embed_data)  # yt^
        flag_first = True
        for block in self.blocks_2:
            if flag_first:  # peek current question
                x = block(mask=1, query=x, key=x,
                          values=x, apply_pos=False, pdiff=pid_embed_data)  # False: 没有FFN, 第一层只有self attention, 对应于xt^
                flag_first = False
            else:  # dont peek current response
                x = block(mask=0, query=x, key=x, values=y, apply_pos=True,
                          pdiff=pid_embed_data)  # True: +FFN+残差+laynorm 非第一层与0~t-1的的q的attention, 对应图中Knowledge Retriever
                # mask=0，不能看到当前的response, 在Knowledge Retrever的value全为0，因此，实现了第一题只有question信息，无qa信息的目的
                # print(x[0,0,:])
                flag_first = True
        return x


class AKT(BaseKt):
    def __init__(self, skill_num, problem_num, emb_size=128, attention_blocks=4, dropout=0.1, d_ff=256, final_fc_dim=256,
                 att_heads=8,
                 l2=1e-5):
        super().__init__("AKT")
        self.skill_num = skill_num
        self.dropout = dropout
        self.problem_num = problem_num
        self.l2 = l2
        self.embed_dim = emb_size
        self.final_fc_dim = final_fc_dim
        self.difficult_param = Embedding(self.problem_num, 1)  # 题目难度
        self.q_embed_diff = Embedding(self.skill_num,
                                      self.embed_dim)  # question emb, 总结了包含当前question（concept）的problems（questions）的变化
        self.qa_embed_diff = Embedding(2 * self.skill_num, self.embed_dim)  # interaction emb, 同上

        self.q_embed_layer = Embedding(self.skill_num, self.embed_dim)
        self.qa_embed_layer = Embedding(2 * self.skill_num, self.embed_dim)  # interaction emb

        # Architecture Object. It contains stack of attention block
        self.arch_model = Architecture(n_blocks=attention_blocks, n_heads=att_heads, dropout=dropout, d_model=self.embed_dim,
                                       d_ff=d_ff, kq_same=1)

        self.out = torch.nn.Sequential(
            Linear(self.embed_dim + self.embed_dim, final_fc_dim), ReLU(), Dropout(self.dropout),
            Linear(final_fc_dim, 256), ReLU(
            ), Dropout(self.dropout),
            Linear(256, 1)
        )

    def forward(self, x, mask=None, training=None, **kwargs):
        qa, q, qp = x
        q_embed = self.q_embed_layer(q)
        qa_embed = self.qa_embed_layer(qa)

        q_embed_diff_data = self.q_embed_diff(q)  # d_ct 总结了包含当前question（concept）的problems（questions）的变化
        pid_embed_data = self.difficult_param(qp)  # uq 当前problem的难度
        q_embed_data = q_embed + pid_embed_data * q_embed_diff_data  # uq *d_ct + c_ct # question encoder

        qa_embed_diff_data = self.qa_embed_diff(qa)  # f_(ct,rt) or #h_rt (qt, rt)差异向量
        qa_embed_data = qa_embed + pid_embed_data * qa_embed_diff_data  # uq* f_(ct,rt) + e_(ct,rt)
        # （q-response emb diffuser + question emb diffuser）
        c_reg_loss = (pid_embed_data ** 2.).sum() * self.l2  # rasch部分loss

        # BS.seqlen,emb_size
        # Pass to the decoder
        # output shape BS,seqlen,emb_size or emb_size//2
        d_output = self.arch_model(q_embed_data, qa_embed_data, pid_embed_data)

        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q).squeeze(-1)
        m = torch.nn.Sigmoid()
        preds = m(output)
        if training:
            self.add_loss("reg_loss",c_reg_loss)
        return preds[:, 1:, None]

    # def train_step(self, data):
    #     x, y, mask, sample_weight = self.data_map(data)
    #     # Compute prediction error
    #     y_pred, c_reg_loss = self(x, training=True, mask=mask)
    #     y_pred, y = y_pred.masked_select(mask), y.masked_select(mask)
    #     loss = self.compute_loss(x, y_pred, y, sample_weight) + c_reg_loss
    #     # Backpropagation
    #     loss.backward()
    #     self.optimizer.step()
    #     self.optimizer.zero_grad()
    #     return self.compute_metrics(x, y_pred, y, sample_weight)

    @property
    def inputs_specs(self):
        return ("skill_response", "skill", "problem"), ("correct",)

    def data_map(self, data):
        (skill_response, skill, problem), y = data
        mask = torch.ge(y, 0).type(torch.int8)
        skill_response = (skill_response * mask).type(torch.long)
        skill = (skill * mask).type(torch.long)
        problem=(problem*mask).type(torch.long)
        y = y.unsqueeze(-1).type(torch.float)
        return (skill_response, skill, problem), y[:, 1:], mask.unsqueeze(-1).type(torch.bool)[:, 1:], None
