#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 2024/8/14 11:26
# @Author: hb925
# @File: wandb_dkt.py

import torch
from torch.nn import Embedding, LSTM, Dropout, Linear
import torch.nn.functional as F
from torch_kt.training import BaseKt


class RevDKT(BaseKt):
    def __init__(self, skill_num, emb_size=128, hidden_units=128, dropout=0.1):
        super().__init__("RevDKT")
        self.num_c = skill_num
        self.emb_size = emb_size
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.interaction_emb = Embedding(self.num_c * 2, self.emb_size)
        self.lstm_layer = LSTM(self.emb_size, self.hidden_units, batch_first=True)
        self.dropout_layer = Dropout(self.dropout)
        self.out_layer = Linear(self.hidden_units, self.num_c)

    def forward(self, x, mask=None, training=None, **kwargs):
        qa, q, r = x
        qa_emb = self.interaction_emb(qa)
        q_onehot = F.one_hot(q, num_classes=self.num_c)
        mask_v = mask.type(torch.float32)
        mask_rev = mask_v.flip(dims=[1])
        # qa_emb = self.emb_layer(qa)
        qa_emb_rev = qa_emb.flip(dims=[1])
        qa_emb_rev = qa_emb_rev * mask_rev
        h_r,_=self.lstm_layer(qa_emb_rev)
        h_r=h_r.flip(dims=[1])
        h, _ = self.lstm_layer(qa_emb)
        h_pre=h[:,:-1,:]
        h_pre = self.dropout_layer(h_pre)
        y = self.out_layer(h_pre)
        y = torch.sigmoid(y)
        y = q_onehot[:, 1:,:] * y
        y=y.sum(dim=-1, keepdim=True)
        # ml = torch.eq(q[:, 1:], q[:, :-1]).to(torch.float32).unsqueeze(dim=-1)
        ml = torch.eq(r[:,1:], torch.tensor([1]).to(r.device)).to(torch.float32)
        pad = torch.zeros([h.shape[0], 1, h.shape[-1]], device=h.device)
        pad_h = torch.cat([pad, h[:, :-1, :]], dim=1)
        pad_h_rev = torch.cat([h_r[:, 1:, :], pad], dim=1)
        other_loss=torch.mean((h_pre-pad_h_rev[:,1:,:])**2*mask_v[:,1:]*ml)

        # other_loss = torch.mean((h_pre - h[:, 1:, :]) ** 2*ml * mask_v[:, 1:])
        if training:
            return y,other_loss
        else:
            return y
    def train_step(self, data):
        x, y, mask, sample_weight = self.data_map(data)
        # Compute prediction error
        mask_n = mask[:, 1:]
        y_pred,other_loss = self(x, training=True, mask=mask)
        y_pred, y = y_pred.masked_select(mask_n), y.masked_select(mask_n)
        loss = self.compute_loss(x, y_pred, y, sample_weight)+other_loss
        # Backpropagation
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return self.compute_metrics(x, y_pred, y, sample_weight)
    @property
    def inputs_specs(self):
        return ("skill_response", "skill"), ("correct",)

    def data_map(self, data):
        (skill_response, skill), y = data
        mask = torch.ge(y, 0).to(torch.int8)
        skill_response = (skill_response * mask).type(torch.long)
        skill = (skill * mask).type(torch.long)
        y = y.unsqueeze(-1).type(torch.float)
        return (skill_response, skill,y), y[:, 1:], mask.unsqueeze(-1).type(torch.bool), None
