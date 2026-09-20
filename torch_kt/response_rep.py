#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/10/16 00:46
# @Author  : hb
# @File    : response_rep.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class JointEmbedding(nn.Module):
    def __init__(self, num_concepts, embedding_dim=64):
        """
        联合嵌入层：将(c_i, a_i)组合视为单一单元
        参数:
            num_concepts: 知识点数量M
            embedding_dim: 嵌入维度d
        """
        super().__init__()
        self.num_concepts = num_concepts
        self.embedding_dim = embedding_dim
        # 创建嵌入矩阵：2M个组合对应d维向量
        self.embedding = nn.Embedding(2 * num_concepts, embedding_dim)

    def forward(self, c, a):
        """
        前向计算
        输入:
            c: 知识点索引 [batch_size]
            a: 答题结果 (0/1) [batch_size]
        输出:
            x: 嵌入向量 [batch_size, embedding_dim]
        """
        # 计算组合索引: index = c_i * 2 + a_i
        indices = c * 2 + a
        return self.embedding(indices)

    def similarity(self, x1, x2):
        """
        计算余弦相似度
        输入:
            x1, x2: 嵌入向量 [batch_size, embedding_dim]
        输出:
            相似度得分 [batch_size]
        """
        return F.cosine_similarity(x1, x2, dim=-1)


class SeparatedEmbedding(nn.Module):
    def __init__(self, num_concepts, embedding_dim=64, mode='additive'):
        """
        分离嵌入层：分别嵌入知识点和答题结果
        参数:
            num_concepts: 知识点数量M
            embedding_dim: 嵌入维度d
            mode: 组合方式 ('additive', 'concat', 'multiplicative')
        """
        super().__init__()
        self.mode = mode
        self.concept_embed = nn.Embedding(num_concepts, embedding_dim)
        self.response_embed = nn.Embedding(2, embedding_dim)  # a_i ∈ {0,1}

        if mode == 'concat':
            self.projection = nn.Linear(2 * embedding_dim, embedding_dim)
        elif mode == 'multiplicative':
            self.variant_embed = nn.Embedding(2 * num_concepts, embedding_dim)

    def forward(self, c, a):
        e_c = self.concept_embed(c)
        e_a = self.response_embed(a)

        if self.mode == 'additive':
            return e_c + e_a
        elif self.mode == 'concat':
            return self.projection(torch.cat([e_c, e_a], dim=-1))
        elif self.mode == 'multiplicative':
            indices = c * 2 + a
            v = self.variant_embed(indices)
            return e_c * v

    def similarity(self, x1, x2):
        return F.cosine_similarity(x1, x2, dim=-1)


class RaschModelEmbedding(nn.Module):
    def __init__(self, num_concepts, num_questions, embedding_dim=64):
        """
        基于Rasch模型的试题-知识点联合嵌入
        参数:
            num_concepts: 知识点数量
            num_questions: 试题数量
            embedding_dim: 嵌入维度d
        """
        super().__init__()
        # 知识点嵌入 c ∈ R^{num_concepts × d}
        self.c_embed = nn.Embedding(num_concepts, embedding_dim)

        # 知识点变体向量 d ∈ R^{num_concepts × d}
        self.d_embed = nn.Embedding(num_concepts, embedding_dim)

        # 试题难度参数 μ ∈ R^{num_questions}
        self.mu = nn.Parameter(torch.zeros(num_questions))

        # 答题结果嵌入 g ∈ R^{2 × d}
        self.g_embed = nn.Embedding(2, embedding_dim)

        # 知识点-答题组合变体向量 f ∈ R^{num_concepts × 2 × d}
        self.f_embed = nn.Embedding(2 * num_concepts, embedding_dim)

    def forward(self, c, q, a):
        """
        前向计算
        输入:
            c: 知识点索引 [batch_size]
            q: 试题索引 [batch_size]
            a: 答题结果 (0/1) [batch_size]
        输出:
            a_t: 交互嵌入向量 [batch_size, embedding_dim]
        """
        # 获取各组件嵌入
        c_vec = self.c_embed(c)  # [batch_size, d]
        d_vec = self.d_embed(c)  # [batch_size, d]
        g_vec = self.g_embed(a)  # [batch_size, d]
        mu_scalar = self.mu[q].unsqueeze(1)  # [batch_size, 1]

        # 计算组合索引: index = c_i * 2 + a_i
        indices = c * 2 + a
        f_vec = self.f_embed(indices)  # [batch_size, d]

        # 计算交互嵌入: a_t = c_{c_t} + g_{r_t} + μ_{q_t} · f_{c_t,r_t}
        return c_vec + g_vec + mu_scalar * f_vec

    def get_question_embedding(self, c, q):
        """
        获取试题嵌入: q_t = c_{c_t} + μ_{q_t} · d_{c_t}
        """
        c_vec = self.c_embed(c)
        d_vec = self.d_embed(c)
        mu_scalar = self.mu[q].unsqueeze(1)
        return c_vec + mu_scalar * d_vec

    def similarity(self, x1, x2):
        return F.cosine_similarity(x1, x2, dim=-1)