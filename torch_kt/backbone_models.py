#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/25 0025 上午 9:29
# @Author  : hb
# @File    : backbone_models.py
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionEmbedding(nn.Module):
    def __init__(self, dim, max_steps=500):
        super().__init__()
        self.register_buffer(
            "embedding", self._build_embedding(dim, max_steps), persistent=False
        )
        self.projection1 = nn.Linear(dim * 2, dim // 2)
        self.projection2 = nn.Linear(dim // 2, dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x


    def _build_embedding(self, dim, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)
        embeddings = math.log(10000) / (dim - 1)
        embeddings = torch.exp(torch.arange(dim) * -embeddings)
        embeddings = steps * embeddings[None, :]
        table = torch.cat([embeddings.sin(), embeddings.cos()], dim=1)
        return table

#######################################

class CondUpsampler(nn.Module):
    def __init__(self, cond_length, target_dim):
        super().__init__()
        self.linear1 = nn.Linear(cond_length, target_dim // 2)
        self.linear2 = nn.Linear(target_dim // 2, target_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.linear2(x)
        x = F.leaky_relu(x, 0.4)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, residual_channels, dilation):
        super().__init__()
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            3,
            padding=dilation,
            dilation=dilation,
            padding_mode="circular",
        )
        self.diffusion_projection = nn.Linear(hidden_size, residual_channels)
        self.conditioner_projection = nn.Conv1d(
            1, 2 * residual_channels, 1, padding=2, padding_mode="circular"
        )
        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)

        nn.init.kaiming_normal_(self.conditioner_projection.weight)
        nn.init.kaiming_normal_(self.output_projection.weight)

    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)

        y = x + diffusion_step
        y = self.dilated_conv(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        x = F.leaky_relu(x, 0.4)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip


class ResidualStart(nn.Module):
    def __init__(
            self, hidden_size, condition_size=-1,
            residual_hidden=64,
            residual_layers=6,
            residual_channels=8,
            dilation_cycle_length=4,
            lambda_uncertainty=0.1, dropout=0.2, diff_steps=1000
    ):
        super().__init__()
        self.dropout = dropout
        self.diff_steps = diff_steps
        self.lambda_uncertainty = lambda_uncertainty
        if condition_size < 0:
            self.condition_size = hidden_size
        else:
            self.condition_size = condition_size
        self.input_projection = nn.Conv1d(
            1, residual_channels, 1, padding=2, padding_mode="circular"
        )
        self.diffusion_embedding = PositionEmbedding(
            residual_hidden, self.diff_steps
        )
        self.cond_upsampler = CondUpsampler(
            target_dim=hidden_size, cond_length=self.condition_size
        )
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(residual_channels=residual_channels,
                              dilation=2 ** (i % dilation_cycle_length),
                              hidden_size=residual_hidden
                              )
                for i in range(residual_layers)
            ]
        )
        # self.layer_normal = nn.LayerNorm(hidden_size)
        self.skip_projection = nn.Conv1d(residual_channels, residual_channels, 3)
        self.output_projection = nn.Conv1d(residual_channels, 1, 3)
        nn.init.kaiming_normal_(self.input_projection.weight)
        nn.init.kaiming_normal_(self.skip_projection.weight)
        # nn.init.zeros_(self.output_projection.weight)

    def forward(self, x_t, t, condition, mask=None):
        x = self.input_projection(x_t)
        if self.lambda_uncertainty > 1e-5:
            lambda_uncertainty = torch.normal(mean=torch.full(x_t.shape, self.lambda_uncertainty),
                                              std=torch.full(x_t.shape, self.lambda_uncertainty)).to(condition.device)
            x = x * lambda_uncertainty
        diffusion_step = self.diffusion_embedding(t)
        cond_up = self.cond_upsampler(condition)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_up, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.tanh(x)
        x = self.output_projection(x)
        # x = self.layer_normal(x)
        return x


##################################

class PositionwiseFeedForward(torch.nn.Module):
    "Implements FFN equation."

    def __init__(self, hidden_units, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(hidden_units, hidden_units * 4)
        self.w_2 = nn.Linear(hidden_units * 4, hidden_units)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_normal_(self.w_1.weight)
        torch.nn.init.xavier_normal_(self.w_2.weight)

    def forward(self, hidden):
        hidden = self.w_1(hidden)
        hidden = self.relu(hidden)
        hidden = self.dropout(hidden)
        hidden = self.w_2(hidden)
        return hidden


class MultiHeadedAttention(torch.nn.Module):
    def __init__(self, heads, hidden_units, dropout):
        super().__init__()
        assert hidden_units % heads == 0
        self.size_head = hidden_units // heads
        self.num_heads = heads
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_units, hidden_units) for _ in range(3)])
        self.w_layer = nn.Linear(hidden_units, hidden_units)
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_normal_(self.w_layer.weight)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        q, k, v = [l(x).view(batch_size, -1, self.num_heads, self.size_head).transpose(1, 2) for l, x in
                   zip(self.linear_layers, (q, k, v))]
        corr = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        if mask is not None:
            mask = mask.unsqueeze(1).repeat([1, corr.shape[1], 1, corr.shape[-1]])
            mask = mask.tril(diagonal=0)
            corr = corr.masked_fill(mask == 0, -1e9)
        else:
            mask = torch.ones_like(corr).to(q.device)
            mask = mask.tril(diagonal=0)
            corr = corr.masked_fill(mask == 0, -1e9)
        prob_attn = F.softmax(corr, dim=-1)
        if self.dropout is not None:
            prob_attn = self.dropout(prob_attn)
        hidden = torch.matmul(prob_attn, v)
        hidden = self.w_layer(hidden.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.size_head))
        return hidden


class TransformerEncoder(torch.nn.Module):
    def __init__(self, hidden_units, attn_heads, dropout):
        super(TransformerEncoder, self).__init__()
        self.attention = MultiHeadedAttention(heads=attn_heads, hidden_units=hidden_units, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_units=hidden_units, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_units)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_units)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout_out = nn.Dropout(p=dropout)

    def forward(self, hidden, mask):
        att_out = self.attention(hidden, hidden, hidden, mask=mask)
        att_out = hidden + self.dropout(att_out)
        att_out = self.norm(att_out)
        out = att_out + self.dropout1(self.feed_forward(att_out))
        out=self.norm1(out)
        out=self.dropout_out(out)
        return out


class DiffuTransformerStart(torch.nn.Module):
    def __init__(self, hidden_units, residual_size=-1, condition_size=-1, blocks_num=4, att_heads=4,
                 lambda_uncertainty=0.1, dropout=0.2,
                 diff_steps=1000):
        super(DiffuTransformerStart, self).__init__()
        self.hidden_size = hidden_units
        if residual_size < 0:
            self.residual_size = hidden_units // 2
        else:
            self.residual_size = residual_size
        if condition_size < 0:
            self.condition_size = hidden_units
        else:
            self.condition_size = condition_size
        self.dropout = dropout
        self.diff_steps = diff_steps
        self.blocks_num = blocks_num
        self.heads = att_heads
        self.dropout_layer = nn.Dropout(self.dropout)
        self.input_projection = nn.Sequential(nn.Linear(hidden_units, self.residual_size),
                                              nn.Tanh())
        self.condition_projection = nn.Sequential(nn.Linear(hidden_units, self.residual_size // 2),
                                                  nn.Tanh(), nn.Linear(self.residual_size // 2, self.residual_size))
        self.time_embed = PositionEmbedding(self.residual_size, self.diff_steps)
        self.step_embed = PositionEmbedding(self.residual_size, 100)

        self.lambda_uncertainty = lambda_uncertainty
        self.transformer_blocks = nn.ModuleList(
            [TransformerEncoder(self.residual_size, self.heads, self.dropout) for _ in range(self.blocks_num)])
        self.w_layer = nn.Sequential(nn.Linear(self.residual_size, self.residual_size * 2),
                                     nn.Tanh(), nn.Linear(self.residual_size * 2, hidden_units))

    def forward(self, x_t, t, condition, mask=None):
        x = self.input_projection(x_t)
        step_embedding = self.time_embed(t)
        step_embedding = step_embedding.unsqueeze(dim=1)
        step_embedding_l = self.step_embed(torch.arange(x_t.shape[1], dtype=torch.long))
        step_embedding_l = step_embedding_l.unsqueeze(dim=0)
        x = x + step_embedding + step_embedding_l
        if self.lambda_uncertainty > 1e-5:
            lambda_uncertainty = torch.normal(mean=torch.full(x.shape, self.lambda_uncertainty),
                                              std=torch.full(x.shape, self.lambda_uncertainty)).to(condition.device)
            x = x * lambda_uncertainty
        condition = self.condition_projection(condition)
        hidden = condition + x
        for transformer in self.transformer_blocks:
            hidden = transformer.forward(hidden, mask)
        hidden = self.w_layer(hidden)
        rep_diffu = self.dropout_layer(hidden)
        return rep_diffu


#####################################################################
class DiffuLstmStart(torch.nn.Module):
    def __init__(self, hidden_units, residual_size=-1, condition_size=-1, lambda_uncertainty=0.1, dropout=0.2,
                 diff_steps=1000):
        super(DiffuLstmStart, self).__init__()
        self.hidden_size = hidden_units
        if residual_size < 0:
            self.residual_size = hidden_units // 2
        else:
            self.residual_size = residual_size
        if condition_size < 0:
            self.condition_size = hidden_units
        else:
            self.condition_size = condition_size
        self.dropout = dropout
        self.diff_steps = diff_steps
        self.lambda_uncertainty = lambda_uncertainty
        self.drop_layer = nn.Dropout(self.dropout)
        self.input_projection = nn.Sequential(nn.Linear(hidden_units, self.residual_size // 2),
                                              nn.Tanh(), nn.Linear(self.residual_size // 2, self.residual_size))
        self.condition_projection = nn.Sequential(nn.Linear(hidden_units, self.residual_size // 2),
                                                  nn.Tanh(), nn.Linear(self.residual_size // 2, self.residual_size))
        self.diffusion_embedding = PositionEmbedding(self.residual_size, self.diff_steps)
        self.att = nn.LSTM(self.residual_size, self.residual_size * 2, batch_first=True)
        self.batch_normal = torch.nn.BatchNorm1d(self.hidden_size)

        self.attention_dim = 80
        self.mlp = nn.Linear(self.residual_size * 2, self.attention_dim)
        self.similarity = nn.Linear(self.attention_dim, 1, bias=False)
        self.fc = nn.Sequential(nn.Linear(self.residual_size * 4, self.residual_size),
                                nn.Tanh(), nn.Linear(self.residual_size, hidden_units))

    def attention_module(self, lstm_output):
        att_w = self.mlp(lstm_output)
        att_w = torch.tanh(att_w)
        att_w = self.similarity(att_w)
        attn_mask = torch.triu(torch.ones(lstm_output.shape[1], lstm_output.shape[1]), diagonal=1).to(
            dtype=torch.bool).to(lstm_output.device)
        att_w = att_w.transpose(1, 2).expand(lstm_output.shape[0], lstm_output.shape[1], lstm_output.shape[1]).clone()
        att_w = att_w.masked_fill_(attn_mask, float("-inf"))
        alphas = torch.nn.functional.softmax(att_w, dim=-1)
        attn_ouput = torch.bmm(alphas, lstm_output)
        # 论文的原始实现没有对未来的att做屏蔽
        # alphas = nn.Softmax(dim=1)(att_w)
        # attn_ouput = alphas * lstm_output  # 整个seq的attn之和为1，计算前面的的时候，所有的attn都<<1，不会有问题？做的少的时候，历史作用小，做得多的时候，历史作用变大？
        attn_output_cum = torch.cumsum(attn_ouput, dim=1)
        attn_output_cum_1 = attn_output_cum - attn_ouput
        final_output = torch.cat((attn_output_cum_1, lstm_output), 2)
        return final_output

    def attention_module1(self, lstm_output, c):
        att_w = self.mlp(lstm_output)
        att_w = torch.tanh(att_w)
        att_w = self.similarity(att_w)
        attn_mask = torch.triu(torch.ones(lstm_output.shape[1], lstm_output.shape[1]), diagonal=1).to(
            dtype=torch.bool).to(lstm_output.device)
        att_w = att_w.transpose(1, 2).expand(lstm_output.shape[0], lstm_output.shape[1], lstm_output.shape[1]).clone()
        att_w = att_w.masked_fill_(attn_mask, float("-inf"))
        # alphas = torch.nn.functional.softmax(att_w, dim=-1)
        # attn_ouput = torch.bmm(alphas, c)
        # 论文的原始实现没有对未来的att做屏蔽
        alphas = torch.nn.functional.softmax(att_w, dim=1)
        attn_ouput = torch.bmm(alphas,
                               c)  # alphas * c  # 整个seq的attn之和为1，计算前面的的时候，所有的attn都<<1，不会有问题？做的少的时候，历史作用小，做得多的时候，历史作用变大？
        attn_output_cum = torch.cumsum(attn_ouput, dim=1)
        attn_output_cum_1 = attn_output_cum - attn_ouput
        final_output = torch.cat((attn_output_cum_1, lstm_output), 2)
        return final_output

    def forward(self, x_t, t, condition, mask=None):
        x = self.input_projection(x_t)
        step_embedding = self.diffusion_embedding(t)
        step_embedding = step_embedding.unsqueeze(dim=1)
        x = x + step_embedding
        if self.lambda_uncertainty > 1e-5:
            lambda_uncertainty = torch.normal(mean=torch.full(x.shape, self.lambda_uncertainty),
                                              std=torch.full(x.shape, self.lambda_uncertainty)).to(condition.device)
            x = x * lambda_uncertainty
        condition = self.condition_projection(condition)
        x_t_o = x + condition
        x_t_o, _ = self.att(x_t_o)
        o = self.attention_module(x_t_o)
        o = self.fc(o)
        o = self.drop_layer(o)
        return o


#############################################


######################################################################
class CausalConv1d(nn.Module):
    """
    Input and output sizes will be the same.
    """

    def __init__(self, in_size, out_size, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_size, out_size, kernel_size, padding=self.pad, dilation=dilation)

    def forward(self, x):
        x = self.conv1(x)
        x = x[..., :-self.pad].contiguous()
        return x


class ZeroConv1d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ZeroConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, x):
        out = self.conv(x)
        return out


class ResidualLayer(nn.Module):
    def __init__(self, residual_size, skip_size, dilation):
        super(ResidualLayer, self).__init__()
        self.conv_filter = CausalConv1d(residual_size, residual_size,
                                        kernel_size=3, dilation=dilation)
        self.conv_gate = CausalConv1d(residual_size, residual_size,
                                      kernel_size=3, dilation=dilation)
        self.resconv1_1 = nn.Conv1d(residual_size, residual_size, kernel_size=1)
        self.skipconv1_1 = nn.Conv1d(residual_size, skip_size, kernel_size=1)

    def forward(self, x):
        conv_filter = self.conv_filter(x)
        conv_gate = self.conv_gate(x)
        fx = F.tanh(conv_filter) * F.sigmoid(conv_gate)
        fx = self.resconv1_1(fx)
        skip = self.skipconv1_1(fx)
        residual = fx + x
        # residual=[batch,residual_size,seq_len]  skip=[batch,skip_size,seq_len]
        return skip, residual


class DilatedStack(nn.Module):
    def __init__(self, residual_size, skip_size, dilation_depth):
        super(DilatedStack, self).__init__()
        residual_stack = [ResidualLayer(residual_size, skip_size, 2 ** layer)
                          for layer in range(dilation_depth)]
        self.residual_stack = nn.ModuleList(residual_stack)

    def forward(self, x):
        skips = []
        for layer in self.residual_stack:
            skip, x = layer(x)
            skips.append(skip.unsqueeze(0))
            # skip =[1,batch,skip_size,seq_len]
        return torch.cat(skips, dim=0), x  #


class TemporalConvStart(nn.Module):

    def __init__(self, hidden_units, residual_size=32, condition_size=-1, dilation_blocks=1, lambda_uncertainty=0.1,
                 dropout=0.2,
                 diff_steps=1000,
                 dilation_depth=6):
        super(TemporalConvStart, self).__init__()

        self.hidden_size = hidden_units
        self.residual_size = residual_size
        self.dropout = dropout
        self.diff_steps = diff_steps
        self.dilation_depth = dilation_depth
        if condition_size < 0:
            self.condition_size = hidden_units
        else:
            self.condition_size = condition_size
        self.lambda_uncertainty = lambda_uncertainty
        self.drop_layer = nn.Dropout(self.dropout)
        self.diffusion_embedding = nn.Sequential(PositionEmbedding(self.residual_size, self.diff_steps),
                                                 nn.Linear(self.residual_size, self.residual_size * 2),
                                                 torch.nn.Tanh(),
                                                 nn.Linear(self.residual_size * 2, self.residual_size))
        self.input_projection = nn.Sequential(nn.Linear(hidden_units, self.residual_size * 3),
                                              torch.nn.Tanh(),
                                              nn.Linear(self.residual_size * 3, self.residual_size))
        self.condition_projection = DilatedStack(self.condition_size, self.residual_size, dilation_depth)
        self.out_conv = nn.Sequential(nn.Linear(self.residual_size, self.hidden_size * 2),
                                      torch.nn.Tanh(),
                                      nn.Linear(self.hidden_size * 2, self.hidden_size))

    def forward(self, x_t, t, condition, mask=None):
        step_embedding = self.diffusion_embedding(t)

        x_t = self.input_projection(x_t)
        step_embedding = step_embedding.unsqueeze(dim=1)

        if self.lambda_uncertainty > 1e-5:
            lambda_uncertainty = torch.normal(mean=torch.full(x_t.shape, self.lambda_uncertainty),
                                              std=torch.full(x_t.shape, self.lambda_uncertainty)).to(x_t.device)
            x_t = x_t * lambda_uncertainty
        x_t = x_t + step_embedding

        condition_p = condition.permute(0, 2, 1)
        condition_p, _ = self.condition_projection(condition_p)
        condition_p = torch.sum(condition_p, dim=0) / math.sqrt(self.dilation_depth)  # [batch,out_size,seq_len]
        condition_p = condition_p.permute(0, 2, 1)
        x_t = x_t * condition_p  # torch.concat([x_t, condition_p], dim=-1)
        out = self.out_conv(condition_p)
        out = self.drop_layer(out)
        return out


class MlpStart(torch.nn.Module):
    def __init__(self, hidden_units, residual_size=32, condition_size=-1, lambda_uncertainty=0.1, dropout=0.2,
                 diff_steps=1000):
        super(MlpStart, self).__init__()

        self.hidden_size = hidden_units
        self.residual_size = residual_size
        self.dropout = dropout
        self.diff_steps = diff_steps
        if condition_size < 0:
            self.condition_size = hidden_units
        else:
            self.condition_size = condition_size
        self.lambda_uncertainty = lambda_uncertainty
        self.drop_layer = torch.nn.Dropout(self.dropout)
        self.diffusion_embedding = torch.nn.Sequential(PositionEmbedding(self.residual_size * 2, self.diff_steps),
                                                       torch.nn.Linear(self.residual_size * 2, self.residual_size),
                                                       torch.nn.Tanh())
        self.input_projection = torch.nn.Sequential(torch.nn.Linear(hidden_units, hidden_units // 3),
                                                    torch.nn.Linear(hidden_units // 3, self.residual_size ),torch.nn.Tanh())
        self.condition_projection = torch.nn.Sequential(torch.nn.Linear(self.condition_size, self.residual_size ), torch.nn.Tanh())
        self.out_conv = torch.nn.Sequential(torch.nn.Linear(self.residual_size, self.residual_size * 2),
                                            torch.nn.Linear(self.residual_size * 2, self.hidden_size), torch.nn.Tanh())
        # self.out_conv = torch.nn.LSTM(self.residual_size * 2, self.hidden_size,batch_first=True)

    def forward(self, x_t, t, condition, mask=None):
        step_embedding = self.diffusion_embedding(t)

        x_t = self.input_projection(x_t)
        step_embedding = step_embedding.unsqueeze(dim=1).repeat(1, x_t.shape[1], 1)

        if self.lambda_uncertainty > 1e-5:
            lambda_uncertainty = torch.normal(mean=torch.full(x_t.shape, self.lambda_uncertainty),
                                              std=torch.full(x_t.shape, self.lambda_uncertainty)).to(x_t.device)
            x_t = x_t * lambda_uncertainty
        x_t = x_t + step_embedding

        condition_p = self.condition_projection(condition)
        x_t =  x_t+ condition_p
        # x_t = torch.concat([step_embedding, condition_p], dim=-1)#x_t+condition_p#torch.concat([x_t, condition_p], dim=-1)
        # out,_=self.out_conv(x_t)
        out = self.out_conv(x_t)
        out = self.drop_layer(out)
        return out
