#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/15 0015 下午 4:20
# @Author  : hb
# @File    : saint.py
import torch
import torch.nn as nn
from torch.nn import Dropout, ReLU
import pandas as pd
from torch.nn import Embedding, Linear
from torch_kt.training import BaseKt


class Encoder_block(nn.Module):
    """
    M = SkipConct(Multihead(LayerNorm(Qin;Kin;Vin)))
    O = SkipConct(FFN(LayerNorm(M)))
    """

    def __init__(self, dim_model, heads_en, total_ex, total_cat, seq_len, dropout=0.1, emb_path="", pretrain_dim=768):
        super().__init__()
        self.seq_len = seq_len
        self.emb_path = emb_path
        self.total_cat = total_cat
        self.total_ex = total_ex
        self.emb_size = dim_model
        self.dropout = dropout
        if total_ex > 0:
            if emb_path == "":
                self.embd_ex = nn.Embedding(total_ex,
                                            embedding_dim=dim_model)  # embedings  q,k,v = E = exercise ID embedding, category embedding, and positionembedding.
            else:
                embs = pd.read_pickle(emb_path)
                self.exercise_embed = Embedding.from_pretrained(embs)
                self.linear = Linear(pretrain_dim, dim_model)
        if total_cat > 0:
            self.emb_cat = nn.Embedding(total_cat, embedding_dim=dim_model)
        # self.embd_pos   = nn.Embedding(max_len, embedding_dim = dim_model)                  #positional embedding

        self.multi_en = nn.MultiheadAttention(embed_dim=dim_model, num_heads=heads_en, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.dropout_layer1 = Dropout(dropout)

        self.ffn_en = torch.nn.Sequential(
            Linear(self.emb_size, self.emb_size),
            ReLU(),
            Dropout(self.dropout),
            Linear(self.emb_size, self.emb_size),
            # Dropout(self.dropout),
        )
        self.layer_norm2 = nn.LayerNorm(dim_model)
        self.dropout_layer2 = Dropout(dropout)

    def forward(self, in_ex, in_cat, in_pos, first_block=True):

        ## todo create a positional encoding (two options numeric, sine)
        if first_block:
            embs = []
            if self.total_ex > 0:
                if self.emb_path == "":
                    in_ex = self.embd_ex(in_ex)
                else:
                    in_ex = self.linear(self.exercise_embed(in_ex))
                embs.append(in_ex)
            if self.total_cat > 0:
                in_cat = self.emb_cat(in_cat)
                embs.append(in_cat)
            out = embs[0]
            for i in range(1, len(embs)):
                out += embs[i]
            out = out + in_pos
            # in_pos = self.embd_pos(in_pos)
        else:
            out = in_ex

        # in_pos = get_pos(self.max_len)
        # in_pos = self.embd_pos(in_pos)

        out = out.permute(1, 0, 2)  # (n,b,d)  # print('pre multi', out.shape)

        # norm -> attn -> drop -> skip corresponging to transformers' norm_first
        # Multihead attention
        n, _, _ = out.shape
        out = self.layer_norm1(out)  # Layer norm
        skip_out = out
        ut_mask = torch.triu(torch.ones(n, n), diagonal=1).to(dtype=torch.bool).to(in_ex.device)
        out, attn_wt = self.multi_en(out, out, out,
                                     attn_mask=ut_mask)  # attention mask upper triangular
        out = self.dropout_layer1(out)
        out = out + skip_out  # skip connection

        # feed forward
        out = out.permute(1, 0, 2)  # (b,n,d)
        out = self.layer_norm2(out)  # Layer norm
        skip_out = out
        out = self.ffn_en(out)
        out = self.dropout_layer2(out)
        out = out + skip_out  # skip connection

        return out


class Decoder_block(nn.Module):
    """
    M1 = SkipConct(Multihead(LayerNorm(Qin;Kin;Vin)))
    M2 = SkipConct(Multihead(LayerNorm(M1;O;O)))
    L = SkipConct(FFN(LayerNorm(M2)))
    """

    def __init__(self, dim_model, total_res, heads_de, seq_len, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.emb_size = dim_model
        self.dropout = dropout
        self.embd_res = nn.Embedding(total_res + 1,
                                     embedding_dim=dim_model)  # response embedding, include a start token
        # self.embd_pos   = nn.Embedding(max_len, embedding_dim = dim_model)                  #positional embedding
        self.multi_de1 = nn.MultiheadAttention(embed_dim=dim_model, num_heads=heads_de,
                                               dropout=dropout)  # M1 multihead for interaction embedding as q k v
        self.multi_de2 = nn.MultiheadAttention(embed_dim=dim_model, num_heads=heads_de,
                                               dropout=dropout)  # M2 multihead for M1 out, encoder out, encoder out as q k v
        self.ffn_en = self.ffn_en = torch.nn.Sequential(
            Linear(self.emb_size, self.emb_size),
            ReLU(),
            Dropout(self.dropout),
            Linear(self.emb_size, self.emb_size),
            # Dropout(self.dropout),
        )  # feed forward layer

        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.layer_norm2 = nn.LayerNorm(dim_model)
        self.layer_norm3 = nn.LayerNorm(dim_model)

        self.dropout_layer1 = Dropout(dropout)
        self.dropout_layer2 = Dropout(dropout)
        self.dropout_layer3 = Dropout(dropout)

    def forward(self, in_res, in_pos, en_out, first_block=True):

        ## todo create a positional encoding (two options numeric, sine)
        if first_block:
            in_in = self.embd_res(in_res)

            # combining the embedings
            out = in_in + in_pos  # (b,n,d)
        else:
            out = in_res

        # in_pos = get_pos(self.max_len)
        # in_pos = self.embd_pos(in_pos)

        out = out.permute(1, 0, 2)  # (n,b,d)# print('pre multi', out.shape)
        n, _, _ = out.shape

        # Multihead attention M1                                     ## todo verify if E to passed as q,k,v
        out = self.layer_norm1(out)
        skip_out = out
        ut_mask = torch.triu(torch.ones(n, n), diagonal=1).to(dtype=torch.bool).to(in_res.device)
        out, attn_wt = self.multi_de1(out, out, out,
                                      attn_mask=ut_mask)  # attention mask upper triangular
        out = self.dropout_layer1(out)
        out = skip_out + out  # skip connection

        # Multihead attention M2                                     ## todo verify if E to passed as q,k,v
        en_out = en_out.permute(1, 0, 2)  # (b,n,d)-->(n,b,d)
        en_out = self.layer_norm2(en_out)
        skip_out = out
        out, attn_wt = self.multi_de2(out, en_out, en_out,
                                      attn_mask=ut_mask)  # attention mask upper triangular
        out = self.dropout_layer2(out)
        out = out + skip_out

        # feed forward
        out = out.permute(1, 0, 2)  # (b,n,d)
        out = self.layer_norm3(out)  # Layer norm
        skip_out = out
        out = self.ffn_en(out)
        out = self.dropout_layer3(out)
        out = out + skip_out  # skip connection

        return out


class SAINT(BaseKt):
    def __init__(self, skill_num, problem_num, max_len, emb_size=128, att_heads=4, dropout=0.1, attention_blocks=4):
        super().__init__("SAINT")
        self.num_q = problem_num
        self.num_c = skill_num
        self.seq_len = max_len
        self.num_en = attention_blocks
        self.num_de = attention_blocks
        self.att_heads = att_heads
        self.emb_size = emb_size
        self.embd_pos = nn.Embedding(max_len, embedding_dim=self.emb_size)

        self.encoder = nn.ModuleList(
            [Encoder_block(self.emb_size, att_heads, self.num_q, self.num_c, self.seq_len, dropout) for _ in
             range(self.num_en)])

        self.decoder = nn.ModuleList(
            [Decoder_block(self.emb_size, 2, att_heads, self.seq_len, dropout) for _ in range(self.num_de)])

        self.dropout_layer = Dropout(dropout)
        self.out = nn.Linear(in_features=self.emb_size, out_features=1)

    def forward(self, x, mask=None, training=None, **kwargs):
        p, q, in_res = x
        batch_size = q.shape[0]
        in_pos = torch.arange(self.seq_len).unsqueeze(0).to(q.device)
        in_pos = self.embd_pos(in_pos)
        first_block = True
        in_ex, in_cat = p, q
        for i in range(self.num_en):
            if i >= 1:
                first_block = False
            in_ex = self.encoder[i](in_ex, in_cat, in_pos, first_block=first_block)
            in_cat = in_ex
        ## pass through each decoder blocks in sequence
        start_token = torch.tensor([[2]]).repeat(batch_size, 1).to(q.device)
        in_res = torch.cat((start_token, in_res), dim=-1)
        # r = in_res
        first_block = True
        for i in range(self.num_de):
            if i >= 1:
                first_block = False
            in_res = self.decoder[i](in_res, in_pos, en_out=in_ex, first_block=first_block)

        ## Output layer

        res = self.out(self.dropout_layer(in_res))
        res = torch.sigmoid(res)
        return res[:, 1:, :]

    @property
    def inputs_specs(self):
        return ("problem", "skill"), ("correct",)

    def data_map(self, data):
        (problem, skill), y = data
        mask = torch.ge(y, 0).to(torch.int8)
        problem = (problem * mask).type(torch.long)
        skill = (skill * mask).type(torch.long)
        y = y.unsqueeze(-1).type(torch.float)
        r = (y[:, :-1] + 1).squeeze(-1).type(torch.long)
        return (problem, skill, r), y[:, 1:], mask.unsqueeze(-1).type(torch.bool)[:, 1:], None
