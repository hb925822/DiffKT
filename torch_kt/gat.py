#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 2026/3/8 11:33
# @Author: hb925
# @File: gat.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """
    标准图注意力层 (Graph Attention Layer)
    输入:
        h: 节点特征矩阵 [N, in_features]
        adj: 邻接矩阵 [N, N] (可以是稠密矩阵，支持广播)
    输出:
        h_prime: 更新后的节点特征 [N, out_features]
    """

    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, n_heads=1, concat=True):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.concat = concat  # 多头是否拼接，如果是最后一层通常为False (取平均)

        # 定义线性变换权重 W (每个头一个)
        # Shape: [n_heads, in_features, out_features]
        self.W = nn.Parameter(torch.empty(size=(n_heads, in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # 定义注意力机制参数 a (每个头一个)
        # 用于计算 e_ij = a^T [Wh_i || Wh_j]
        # Shape: [n_heads, 2 * out_features, 1]
        self.a = nn.Parameter(torch.empty(size=(n_heads, 2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, h, adj):
        """
        h: [N, in_features]
        adj: [N, N] (0/1 矩阵或加权矩阵)
        """
        N = h.size(0)

        # 1. 线性变换: Wh
        # h: [N, in] -> Wh: [N, heads, out]
        Wh = torch.matmul(h, self.W)

        # 2. 计算注意力系数 e
        # 我们需要计算每对节点 (i, j) 的注意力
        # Wh_i: [N, heads, out], Wh_j: [N, heads, out]
        # 构造 [Wh_i || Wh_j]: [N, N, heads, 2*out]

        Wh_i = Wh.unsqueeze(1).expand(-1, N, -1, -1)  # [N, N, heads, out]
        Wh_j = Wh.unsqueeze(0).expand(N, -1, -1, -1)  # [N, N, heads, out]

        Wh_concat = torch.cat([Wh_i, Wh_j], dim=-1)  # [N, N, heads, 2*out]

        # 应用注意力向量 a
        # a: [heads, 2*out, 1] -> 需要调整为 [1, 1, heads, 2*out, 1] 以便广播
        # Wh_concat: [N, N, heads, 2*out] -> 调整为 [N, N, heads, 2*out, 1]
        Wh_concat = Wh_concat.unsqueeze(-1)
        a_expanded = self.a.unsqueeze(0).unsqueeze(0)  # [1, 1, heads, 2*out, 1]

        e = torch.matmul(Wh_concat, a_expanded).squeeze(-1)  # [N, N, heads]
        e = e.permute(2, 0, 1)  # [heads, N, N]

        # 3. 掩码 (Masking): 只保留图中存在的边
        # adj: [N, N] -> [1, N, N]
        mask = adj.unsqueeze(0)
        e = e.masked_fill(mask == 0, -1e9)  # 不存在的边设为负无穷

        # 4. LeakyReLU 和 Softmax
        e = self.leakyrelu(e)
        attention = F.softmax(e, dim=-1)  # 对列 (j) 进行 softmax, dim=-1 对应 N (邻居)

        # Dropout on attention coefficients
        attention = F.dropout(attention, self.dropout, training=self.training)

        # 5. 加权聚合
        # Wh: [N, heads, out] -> [heads, N, out]
        Wh_perm = Wh.permute(1, 0, 2)
        # h_prime: [heads, N, out] = [heads, N, N] @ [heads, N, out]
        h_prime = torch.matmul(attention, Wh_perm)

        # 6. 多头处理
        if self.concat:
            # [heads, N, out] -> [N, heads * out]
            return h_prime.permute(1, 0, 2).contiguous().view(N, -1)
        else:
            # [heads, N, out] -> [N, out] (取平均)
            return h_prime.mean(dim=0)


class EGATLayer(nn.Module):
    """
    边增强图注意力层 (Edge-Enhanced Graph Attention Layer)
    专为 MSKT 设计：利用题目-概念权重或概念图关系作为边特征 e_ij

    输入:
        h: 节点特征 [N, in_features]
        adj: 邻接矩阵 [N, N] (结构掩码)
        edge_feat: 边特征矩阵 [N, N, edge_in_features]
                   (在 MSKT 中，这可能是 C_t^H 的行向量扩展，或概念图的权重矩阵)
    输出:
        h_prime: [N, out_features]
    """

    def __init__(self, in_features, out_features, edge_in_features, dropout=0.6, alpha=0.2, n_heads=1, concat=True):
        super(EGATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.edge_in_features = edge_in_features
        self.n_heads = n_heads
        self.concat = concat

        # 1. 节点特征变换 W
        self.W = nn.Parameter(torch.empty(size=(n_heads, in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # 2. 边特征变换 W_e (将边特征映射到与节点特征相同的空间，以便融合)
        # 或者直接将边特征拼接到注意力计算中
        # 这里采用：将边特征通过 MLP 映射到 attention score 的空间
        # 策略：e_ij_score = LeakyReLU( a^T [Wh_i || Wh_j || MLP(e_ij)] )

        self.W_edge = nn.Parameter(torch.empty(size=(n_heads, edge_in_features, out_features)))
        nn.init.xavier_uniform_(self.W_edge.data, gain=1.414)

        # 注意力参数 a: 输入维度 = out (node_i) + out (node_j) + out (edge)
        total_att_dim = 3 * out_features
        self.a = nn.Parameter(torch.empty(size=(n_heads, total_att_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, h, adj, edge_feat):
        """
        h: [N, in_features]
        adj: [N, N] (结构掩码)
        edge_feat: [N, N, edge_in_features] (边特征，如果某条边不存在，其特征应为0或忽略)
        """
        N = h.size(0)

        # 1. 节点线性变换 Wh: [N, heads, out]
        Wh = torch.matmul(h, self.W)

        # 2. 边特征线性变换 We: [N, N, heads, out]
        # edge_feat: [N, N, edge_in]
        We = torch.matmul(edge_feat, self.W_edge)  # [N, N, heads, out]
        We = We.permute(2, 0, 1, 3)  # [heads, N, N, out]

        # 3. 构造注意力输入 [Wh_i || Wh_j || We_ij]
        Wh_i = Wh.unsqueeze(1).expand(-1, N, -1, -1)  # [N, N, heads, out]
        Wh_j = Wh.unsqueeze(0).expand(N, -1, -1, -1)  # [N, N, heads, out]

        # 调整维度以匹配 We [heads, N, N, out]
        Wh_i = Wh_i.permute(2, 0, 1, 3)  # [heads, N, N, out]
        Wh_j = Wh_j.permute(2, 0, 1, 3)  # [heads, N, N, out]

        concat_all = torch.cat([Wh_i, Wh_j, We], dim=-1)  # [heads, N, N, 3*out]
        concat_all = concat_all.unsqueeze(-1)  # [heads, N, N, 3*out, 1]

        # 应用注意力向量 a
        a_expanded = self.a.unsqueeze(1).unsqueeze(2)  # [heads, 1, 1, 3*out, 1]
        e = torch.matmul(concat_all, a_expanded).squeeze(-1)  # [heads, N, N]

        # 4. 掩码
        mask = adj.unsqueeze(0)  # [1, N, N]
        e = e.masked_fill(mask == 0, -1e9)

        # 5. LeakyReLU + Softmax
        e = self.leakyrelu(e)
        attention = F.softmax(e, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # 6. 加权聚合 (只聚合节点特征 Wh，边特征仅用于计算权重)
        Wh_perm = Wh.permute(1, 0, 2)  # [heads, N, out]
        h_prime = torch.matmul(attention, Wh_perm)  # [heads, N, out]

        # 7. 多头处理
        if self.concat:
            return h_prime.permute(1, 0, 2).contiguous().view(N, -1)
        else:
            return h_prime.mean(dim=0)


# ==========================================
# 示例用法 (模拟 MSKT 场景)
# ==========================================

if __name__ == "__main__":
    # 假设参数
    N_concepts = 50  # 概念数量 (N)
    d_node = 64  # 节点隐藏维度 (d)
    d_edge = 32  # 边特征维度 (例如题目考察权重向量的维度)
    n_heads = 4

    # 1. 构造输入数据
    # 学生概念状态矩阵 S_t (Node Features)
    h = torch.randn(N_concepts, d_node)

    # 概念图邻接矩阵 (Adjacency Matrix) - 假设有向图
    # 这里随机生成一个稀疏矩阵作为示例
    adj = torch.randint(0, 2, (N_concepts, N_concepts)).float()
    torch.fill_diagonal(adj, 1)  # 自环

    # 边特征矩阵 (Edge Features)
    # 在 MSKT 中，这可以是 C_t^H 的某种变换，或者概念间的先验权重
    # Shape: [N, N, d_edge]
    edge_feat = torch.randn(N_concepts, N_concepts, d_edge)
    # 将不存在的边的特征置零 (可选，取决于具体实现逻辑)
    edge_feat = edge_feat * adj.unsqueeze(-1)

    # 2. 实例化模型
    gat_layer = GATLayer(in_features=d_node, out_features=64, n_heads=n_heads, concat=True)
    egat_layer = EGATLayer(in_features=d_node, out_features=64, edge_in_features=d_edge, n_heads=n_heads, concat=True)

    # 3. 前向传播
    print("Input Shape:", h.shape)

    # GAT 输出
    out_gat = gat_layer(h, adj)
    print("GAT Output Shape:", out_gat.shape)  # 应为 [N, heads * 64]

    # EGAT 输出
    out_egat = egat_layer(h, adj, edge_feat)
    print("EGAT Output Shape:", out_egat.shape)  # 应为 [N, heads * 64]


    # 4. 堆叠多层 (模拟 MSKT 中的 Diffusion Generation 部分)
    class MSKT_GNN_Block(nn.Module):
        def __init__(self, in_dim, hidden_dim, edge_dim):
            super().__init__()
            self.gat1 = GATLayer(in_dim, hidden_dim, n_heads=4, concat=True)
            self.bn1 = nn.BatchNorm1d(hidden_dim * 4)
            self.egat = EGATLayer(hidden_dim * 4, hidden_dim, edge_dim, n_heads=4, concat=False)  # 最后一层不拼接
            self.bn2 = nn.BatchNorm1d(hidden_dim)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.6)

        def forward(self, h, adj, edge_feat):
            x = self.gat1(h, adj)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.dropout(x)

            x = self.egat(x, adj, edge_feat)
            x = self.bn2(x)
            x = self.relu(x)
            return x


    model = MSKT_GNN_Block(d_node, 64, d_edge)
    final_out = model(h, adj, edge_feat)
    print("Final Output Shape:", final_out.shape)  # [N, 64]