#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 2024/8/29 18:45
# @Author: hb925
# @File: training.py.py
import numpy as np
import torch
from sklearn import metrics

import torch.nn.functional as F


class BaseKt(torch.nn.Module):
    def __init__(self, name, **kwargs):
        if len(kwargs) > 0:
            print(f"unused params for BaseKt model:{kwargs}")
        super().__init__()
        self.name = name
        self.optimizer = None
        self._losses = []
        self._ext_losses_tensor = {}
        self._ext_losses = {}
        self._labels = []
        self._outputs = []

    def forward(self, x, mask=None, training=None, **kwargs):
        raise NotImplementedError

    def add_loss(self, name, value):
        # self._ext_losses_tensor.setdefault(name, value) 反向传播报错
        self._ext_losses_tensor[name] = value
        # if name not in self._ext_losses.keys():
        #     self._ext_losses[name] = []
        # self._ext_losses[name].append(value)
    @property
    def all_losses(self):
        return self._losses

    @property
    def all_preds(self):
        return self._outputs

    @property
    def all_labels(self):
        return self._labels
    def compile_model(self, optimizer=None, lr=0.001, weight_decay=0):
        if str.lower(optimizer) == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif str.lower(optimizer) == "sgd":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError("unknow optimizer name")

    def reset_state(self):
        self._losses.clear()
        self._labels.clear()
        self._outputs.clear()
        self._ext_losses.clear()
    def predict(self, x, mask=None,  **kwargs):
        pass
    def predict_all_skills(self, x, mask=None, **kwargs):
        pass
    # def compute_loss(self, x, y_pred, y, sample_weight=None):
    #     loss = F.binary_cross_entropy(y_pred, y)
    #     for k, v in self._ext_losses_tensor.items():
    #         loss = loss+v
    #         if k not in self._ext_losses.keys():
    #             self._ext_losses[k] = []
    #         self._ext_losses[k].append(v.detach().cpu().item())
    #     self._losses.append(loss.detach().cpu().item())
    #     return loss
    #
    # def compute_metrics(self, x, y_pred, y, sample_weight):
    #     y_pred, y = y_pred.detach().cpu().numpy(), y.detach().cpu().numpy()
    #     self._labels.append(y)
    #     self._outputs.append(y_pred)
    #     loss = np.mean(self._losses)
    #     return {"loss": loss}
    #
    # def metric_result(self):
    #     metrics_dict = {}
    #     loss = np.mean(self._losses)
    #     metrics_dict.update({"loss": loss})
    #     for k, v in self._ext_losses.items():
    #         metrics_dict.update({k: np.mean(v)})
    #     ts = np.concatenate(self._labels, axis=0)
    #     ps = np.concatenate(self._outputs, axis=0)
    #     prelabels = [1 if p >= 0.5 else 0 for p in ps]
    #     auc = metrics.roc_auc_score(ts, ps)
    #     acc = metrics.accuracy_score(ts, prelabels)
    #     metrics_dict.update({"auc": auc, "acc": acc})
    #     return metrics_dict

    def compute_loss(self, x, y_pred, y, sample_weight=None):
        loss = F.binary_cross_entropy(y_pred, y)
        for k, v in self._ext_losses_tensor.items():
            loss = loss+v
            if k not in self._ext_losses.keys():
                self._ext_losses[k] = []
            self._ext_losses[k].append(v.detach())
        self._losses.append(loss.detach())
        return loss

    def compute_metrics(self, x, y_pred, y, sample_weight):
        y_pred, y = y_pred.detach(), y.detach()
        self._labels.append(y)
        self._outputs.append(y_pred)
        loss = torch.stack(self._losses).mean().cpu().item()
        return {"loss": loss}

    def metric_result(self):
        metrics_dict = {}
        loss = torch.stack(self._losses).mean().cpu().item()
        metrics_dict.update({"loss": loss})
        for k, v in self._ext_losses.items():
            metrics_dict.update({k: torch.stack(v).mean().cpu().item()})
        ts = torch.cat(self._labels, dim=0).cpu().numpy()
        ps = torch.cat(self._outputs, dim=0).cpu().numpy()
        prelabels = [1 if p >= 0.5 else 0 for p in ps]
        auc = metrics.roc_auc_score(ts, ps)
        acc = metrics.accuracy_score(ts, prelabels)
        rmse = metrics.mean_squared_error(ts, ps)
        rmse=np.sqrt(rmse)
        metrics_dict.update({"auc": auc, "acc": acc,"rmse":rmse})
        return metrics_dict

    def train_step(self, data):
        x, y, mask, sample_weight = self.data_map(data)
        # Compute prediction error
        outputs = self(x, training=True, mask=mask)
        if isinstance(outputs, (list, tuple)):
            y_pred = outputs[0]
        else:
            y_pred = outputs
        if y.shape[1] < mask.shape[1]:
            mask = mask[:, 1:]
        y_pred, y = y_pred.masked_select(mask), y.masked_select(mask)
        loss = self.compute_loss(x, y_pred, y, sample_weight)
        # Backpropagation
        # 清空梯度
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return self.compute_metrics(x, y_pred, y, sample_weight)

    def test_step(self, data):
        x, y, mask, sample_weight = self.data_map(data)
        # Compute prediction error
        outputs = self(x, training=False, mask=mask)
        if isinstance(outputs, (list, tuple)):
            y_pred = outputs[0]
        else:
            y_pred = outputs
        if y.shape[1] < mask.shape[1]:
            mask = mask[:, 1:]
        y_pred, y = y_pred.masked_select(mask), y.masked_select(mask)
        self.compute_loss(x, y_pred, y, sample_weight)
        return self.compute_metrics(x, y_pred, y, sample_weight)

    @property
    def inputs_specs(self):
        return ("skill_response", "skill"), "correct"

    def data_map(self, data):
        raise NotImplementedError
