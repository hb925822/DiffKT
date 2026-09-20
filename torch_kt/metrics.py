#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 2024/8/29 20:40
# @Author: hb925
# @File: metrics.py
import numpy as np
from sklearn import metrics


class BinaryAcc(object):
    def __init__(self, name=None):
        super().__init__()
        self.name = name
        self.y_trues = []
        self.y_preds = []
        self.cal_fn = metrics.accuracy_score

    def update_state(self, y_true, y_pred, sample_weight=None):
        pre_labels = [1 if p >= 0.5 else 0 for p in y_pred]
        self.y_trues.append(y_true)
        self.y_preds.append(pre_labels)

    def result(self):
        ts = np.concatenate(self.y_trues, axis=0)
        ps = np.concatenate(self.y_preds, axis=0)
        v = self.cal_fn(ts, ps)
        return v

    def reset_state(self):
        self.y_trues = []
        self.y_preds = []


class BinaryAuc(object):
    def __init__(self, name=None):
        super().__init__()
        self.name = name
        self.y_trues = []
        self.y_preds = []
        self.cal_fn = metrics.roc_auc_score

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.y_trues.append(y_true)
        self.y_preds.append(y_pred)

    def result(self):
        ts = np.concatenate(self.y_trues, axis=0)
        ps = np.concatenate(self.y_preds, axis=0)
        v = self.cal_fn(ts, ps)
        return v

    def reset_state(self):
        self.y_trues.clear()
        self.y_preds.clear()


class MeanMetric(object):
    def __init__(self, name=None):
        super().__init__()
        self.name = name
        self.all_values = []

    def update_state(self, v, sample_weight=None):
        self.all_values.append(v)

    def result(self):
        return np.mean(self.all_values)

    def reset_state(self):
        self.all_values.clear()


class BinaryAcc(object):
    def __init__(self, name=None):
        super().__init__()
        self.name = name
        self.y_trues = []
        self.y_preds = []
        self.cal_fn = metrics.accuracy_score

    def update_state(self, y_true, y_pred, sample_weight=None):
        prelabels = [1 if p >= 0.5 else 0 for p in y_pred]
        self.y_trues.append(y_true)
        self.y_preds.append(prelabels)

    def result(self):
        ts = np.concatenate(self.y_trues, axis=0)
        ps = np.concatenate(self.y_preds, axis=0)
        v = self.cal_fn(ts, ps)
        return v

    def reset_state(self):
        self.y_trues = []
        self.y_preds = []


# class BinaryAuc(object):
#     def __init__(self, name=None):
#         super().__init__()
#         self.name = name
#         self.y_trues = []
#         self.y_preds = []
#         self.cal_fn = metrics.roc_auc_score
#
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         self.y_trues.append(y_true)
#         self.y_preds.append(y_pred)
#
#     def result(self):
#         ts = np.concatenate(self.y_trues, axis=0)
#         ps = np.concatenate(self.y_preds, axis=0)
#         v = self.cal_fn(ts, ps)
#         return v
#
#     def reset_state(self):
#         self.y_trues.clear()
#         self.y_preds.clear()
#
#
# class MeanMetric(object):
#     def __init__(self, name=None):
#         super().__init__()
#         self.name = name
#         self.all_loss = []
#
#     def update_state(self, v, sample_weight=None):
#         self.all_loss.append(v)
#
#     def result(self):
#         return np.mean(self.all_loss)
#
#     def reset_state(self):
#         self.all_loss.clear()
