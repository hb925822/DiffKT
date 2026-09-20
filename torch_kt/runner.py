#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 2024/7/19 13:47
# @Author: hb925
# @File: runner.py
import argparse
import copy
import datetime
import importlib
import inspect
import json
import os
import random
import shutil
import sys
import time
import traceback
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from torch_kt.data_loader import DirDataset
from torch_kt.utils import root_dir


def _import_model_module(model_name):
    if isinstance(model_name, str):
        model_module_name = model_name.lower().replace("-", "_")
        if model_module_name in sys.modules:
            model_module = sys.modules[model_module_name]
        else:
            try:
                model_module = importlib.import_module(name="." + model_module_name, package="torch_kt.models")
            except Exception as e:
                raise ModuleNotFoundError(
                    f'未找到{model_name}对应的模块{e}')  # model_module = get_model_module.__module__  #
    else:
        raise ValueError(f'model_name only can be str or tf.keras.Model')
    return model_module


def get_model_default_params(model_name):
    model_module = _import_model_module(model_name)
    model_module_cls = model_module.__dict__[model_name.replace("-", "")]
    default_model_config = {}
    parms = inspect.signature(model_module_cls.__init__).parameters
    for pn, pt in parms.items():
        if pt.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and pt.default != inspect._empty:
            default_model_config[pn] = pt.default
    return default_model_config


def get_model(model_name, data_name, model_config):
    model_module = _import_model_module(model_name)
    model_module_cls = model_module.__dict__[model_name.replace("-", "")]
    best_model_config = {}
    best_model_config.update(model_config)
    key_args = {}
    args = []
    parms = inspect.signature(model_module_cls.__init__).parameters
    for pn, pt in parms.items():
        if pt.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and pn in best_model_config.keys():
            key_args[pn] = best_model_config[pn]
        if pt.kind == inspect.Parameter.POSITIONAL_ONLY and pn in best_model_config.keys():
            args.append(best_model_config[pn])
    model = model_module_cls(*args, **key_args)
    assert model.name == model_name
    return model


def unpack_x_y_sample_weight(data):
    if isinstance(data, list):
        data = tuple(data)
    if not isinstance(data, tuple):
        return (data, None, None)
    elif len(data) == 1:
        return (data[0], None, None)
    elif len(data) == 2:
        return (data[0], data[1], None)
    elif len(data) == 3:
        return (data[0], data[1], data[2], None)
    else:
        error_msg = (
            "Data is expected to be in format `x`, `(x,)`, `(x, y)`, "
            "or `(x, y, sample_weight)`, found: {}"
        ).format(data)
        raise ValueError(error_msg)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def logs2str(logs: Dict) -> str:
    logstrs = []
    for k, v in logs.items():
        logstrs.append(f'{k}:{v:.4f}')
    return ",".join(logstrs)


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    elif isinstance(value, (tuple, list)):
        vv = [to_device(_, device) for _ in value]
        return vv
    elif isinstance(value, dict):
        newValue = {}
        for k, v in value.items():
            newValue[k] = to_device(v, device)
        return newValue
    else:
        raise ValueError("nonsupport type!")


class KtTrainner(object):
    def __init__(self, log_dir=None, max_epochs=1,
                 device="cuda", patience=10, valid_interval=1, monitor="auc"):
        self.device = device
        self.log_dir = log_dir
        self.max_epochs = max_epochs
        self.patience = patience
        self.valid_interval = valid_interval
        self.best_monitor = monitor

    def rest_state(self, model, training=False):
        model.to(self.device)
        model.reset_state()
        if training:
            model.train()
        else:
            model.eval()

    def _should_stop(self, epoch, best_epoch):
        if 0 < self.patience * self.valid_interval < (epoch - best_epoch):
            return True
        else:
            return False

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of an epoch.

        Subclasses should override for any actions to run. This function should
        only be called during TRAIN mode.

        Args:
            epoch: Integer, index of epoch.
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch.

        Subclasses should override for any actions to run. This function should
        only be called during TRAIN mode.

        Args:
            epoch: Integer, index of epoch.
            logs: Dict, metric results for this training epoch, and for the
              validation epoch if validation is performed. Validation result
              keys are prefixed with `val_`. For training epoch, the values of
              the `Model`'s metrics are returned. Example:
              `{'loss': 0.2, 'accuracy': 0.7}`.
        """

    def on_train_batch_begin(self, batch, logs=None):
        """Called at the beginning of a training batch in `fit` methods.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be called every
        `N` batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """
        # For backwards compatibility.

    def on_train_batch_end(self, batch, logs=None):
        """Called at the end of a training batch in `fit` methods.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be called every
        `N` batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Aggregated metric results up until this batch.
        """
        # For backwards compatibility.

    def on_test_batch_begin(self, batch, logs=None):
        """Called at the beginning of a batch in `evaluate` methods.

        Also called at the beginning of a validation batch in the `fit`
        methods, if validation data is provided.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be called every
        `N` batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """

    def on_test_batch_end(self, batch, logs=None):
        """Called at the end of a batch in `evaluate` methods.

        Also called at the end of a validation batch in the `fit`
        methods, if validation data is provided.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be called every
        `N` batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Aggregated metric results up until this batch.
        """

    def on_predict_batch_begin(self, batch, logs=None):
        """Called at the beginning of a batch in `predict` methods.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be called every
        `N` batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """

    def on_predict_batch_end(self, batch, logs=None):
        """Called at the end of a batch in `predict` methods.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be called every
        `N` batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Aggregated metric results up until this batch.
        """

    def on_train_begin(self, logs=None):
        """Called at the beginning of training.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """

    def on_train_end(self, logs=None):
        """Called at the end of training.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently the output of the last call to
              `on_epoch_end()` is passed to this argument for this method but
              that may change in the future.
        """

    def on_test_begin(self, logs=None):
        """Called at the beginning of evaluation or validation.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """

    def on_test_end(self, logs=None):
        """Called at the end of evaluation or validation.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently the output of the last call to
              `on_test_batch_end()` is passed to this argument for this method
              but that may change in the future.
        """

    def on_predict_begin(self, logs=None):
        """Called at the beginning of prediction.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """

    def on_predict_end(self, logs=None):
        """Called at the end of prediction.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """

    def fit(self, model, train_data, test_data, val_data=None,clean_weights=False, **kwargs):
        if len(kwargs) > 0:
            print(f"unused params for train:{kwargs}")
        model.to(self.device)
        self.rest_state(model, training=True)
        self.on_train_begin()
        total_params_num=sum(p.numel() for p in model.parameters() if p.requires_grad)
        os.makedirs(os.path.join(self.log_dir, "weights"), exist_ok=True)
        weight_path = os.path.join(self.log_dir, "weights", 'model.pth')
        csv_log_path = os.path.join(self.log_dir, "train.csv")
        test_log_path = os.path.join(self.log_dir, "test_result.csv")
        summary_log_path = os.path.join(self.log_dir, "summary.json")
        train_logs_data = []
        if val_data is None:
            val_data = test_data
        total_batch = None
        progressbar_str = ""
        max_auc = 0
        best_epoch = -1
        width = 30
        tarin_setp_time=0.0
        for epoch in range(self.max_epochs):
            model.train()
            model.reset_state()
            self.on_epoch_begin(epoch)
            print(f"Epoch {epoch + 1}/{self.max_epochs}")
            last_train_out_str = ""
            enpoch_start = time.time()
            batch = 0
            logs = {}
            val_logs = {}
            step_times = []
            for batch, data in enumerate(train_data):
                self.on_train_batch_begin(batch)
                step_start = time.time()
                data = to_device(data, self.device)
                tmp_logs = model.train_step(data)
                logs = tmp_logs
                now = time.time()
                time_step = round((now - step_start) * 1000)
                step_times.append(time_step)
                progressbar_str = ""
                if total_batch is None:
                    bar = f"{batch + 1:7d}/unknown,[{'=' * width}]"
                    progressbar_str += bar
                else:
                    numdigits = int(np.log10(total_batch)) + 1
                    prog_width = round((batch + 1) / total_batch * width)
                    bar = f"{batch + 1:{numdigits}d}/{total_batch},[{'=' * prog_width}{'.' * round(width - prog_width)}]"
                    progressbar_str += bar
                if total_batch is not None and batch + 1 == total_batch:
                    progressbar_str += f"- {round(now - enpoch_start):^3d}s :{round(np.mean(step_times))}ms/step,"
                else:
                    progressbar_str += f"- ETA :{time_step}ms/step,"
                train_out_str_add = progressbar_str + logs2str(logs)
                print(len(last_train_out_str) * "\b" + "\r" + train_out_str_add, end="")
                last_train_out_str = train_out_str_add
                self.on_train_batch_end(batch, logs=logs)
            tarin_setp_time=round(np.mean(step_times),3)
            tr_logs = model.metric_result()
            tr_logs = {
                "train_" + name: val for name, val in tr_logs.items()
            }
            trlogs_str = logs2str(tr_logs)
            print(len(last_train_out_str) * "\b" + "\r" + progressbar_str + trlogs_str, end="")
            if total_batch is None:
                total_batch = batch + 1
            if epoch % self.valid_interval == 0:
                val_logs = self.evaluate(model, val_data)
                if "auc" in val_logs.keys() and val_logs["auc"] - max_auc > 0.0001:
                    max_auc = val_logs["auc"]
                    best_epoch = epoch
                    torch.save(model.state_dict(), weight_path)
                val_logs = {
                    "val_" + name: val for name, val in val_logs.items()
                }
            print("\r" + progressbar_str + trlogs_str + "," + logs2str(val_logs))
            train_log_record = {"enpoch": epoch}
            train_log_record.update(tr_logs)
            train_log_record.update(val_logs)
            train_logs_data.append(train_log_record)
            enpoch_logs = copy.copy(tr_logs)
            enpoch_logs.update(val_logs)
            enpoch_logs.update({"enpoch": epoch})
            self.on_epoch_end(epoch, logs=enpoch_logs)
            if self._should_stop(epoch, best_epoch):
                break
        self.on_train_end()
        pd.DataFrame(train_logs_data).to_csv(csv_log_path, index=False, encoding='utf-8',
                                             float_format='%.5f')
        if os.path.exists(weight_path):
            stat_dict = torch.load(weight_path, weights_only=True)
            model.load_state_dict(stat_dict)
        test_start = time.time()
        test_logs = self.evaluate(model, test_data)
        test_total_time = round((time.time() - test_start) * 1000)
        infer_step_time=round(test_total_time/len(test_data),3)
        self.on_test_end(logs=test_logs)
        test_logs1 = {
            "test_" + name: val for name, val in test_logs.items()
        }
        print(logs2str(test_logs1))
        pd.DataFrame([test_logs]).to_csv(test_log_path, index=False, encoding='utf-8',
                                         float_format='%.5f')
        summary_data = {"param_nums":total_params_num,"tarin_setp_time": tarin_setp_time, "infer_step_time":infer_step_time}
        with open(summary_log_path,"w",encoding='utf-8') as f:
            json.dump(summary_data,f)
        if clean_weights:
            shutil.rmtree(os.path.join(self.log_dir, "weights"))
        return test_logs

    def evaluate(self, model, data_set):
        self.rest_state(model, training=False)
        with torch.no_grad():
            self.on_test_begin()
            for step, data in enumerate(data_set):
                data = to_device(data, self.device)
                model.test_step(data)
        logs = model.metric_result()
        self.on_test_end()
        return logs


def train_loop(model, dataname, datas_dir, log_dir, max_len, batch_size=32, max_epochs=50, lr=0.01, optimizer="adam",
               patience=5, valid_interval=1, device="cuda", weight_decay=0, is_cross_validate=False,fast_run=False, **kwargs):
    feature_names, label_names = model.inputs_specs
    trdata_loader, vadata_loader, tedata_loader = dataname.split_generator(datas_dir, feature_names,
                                                                           label_names,sample_num=256 if fast_run else -1,
                                                                           max_len=max_len)
    train_dataloader = DataLoader(trdata_loader, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(vadata_loader, batch_size=batch_size)
    if is_cross_validate:
        test_dataloader = val_dataloader
    else:
        test_dataloader = DataLoader(tedata_loader, batch_size=batch_size)
    model.compile_model(optimizer=optimizer, lr=lr, weight_decay=weight_decay)
    trainner = KtTrainner(log_dir, max_epochs,
                          device=device, patience=patience, valid_interval=valid_interval)
    test_logs = trainner.fit(model,
                             train_dataloader,
                             test_dataloader,
                             val_dataloader,
                             **kwargs)
    return test_logs


def run_preprocess(model_name, data_name, data_base, logs_base, max_len, folds, fast_run=False, **kwargs):
    if data_base is None:
        data_base = os.path.join(root_dir(), "data")
    if logs_base is None:
        logs_base = os.path.join(root_dir(), "logs")
    dataset_dir = os.path.join(data_base, data_name)
    data_set = DirDataset(dataset_dir)
    assert data_name == data_set.name
    data_set.recheck_data(folds=folds, max_len=max_len)
    log_path = f'{data_name}_{model_name}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'
    log_dir = os.path.join(logs_base, log_path)
    os.makedirs(log_dir, exist_ok=True)
    if folds > 0:
        folds_params = [(os.path.join(dataset_dir, f'k{k}'), os.path.join(log_dir, f'k{k}')) for k in range(folds)]
    else:
        folds_params = [(dataset_dir, log_dir)]
    model_config = get_model_default_params(model_name)
    model_config.update(data_set.params)
    if os.path.exists(os.path.join(root_dir(), "best_config.json")):
        with open(os.path.join(root_dir(), "best_config.json"), "r", encoding="utf-8") as f:
            best_model_params = json.load(f)
            best_params = best_model_params.get(model_name, {})
            model_config.update(best_params.get(data_name, {}))
    model_config.update(kwargs)
    model_config.update({"max_len": max_len})
    run_config = model_config.copy()
    run_config.update({"model_name": model_name, "data_name": data_name, "max_len": max_len, "folds": folds})
    with open(os.path.join(log_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=4)
    return folds_params, log_dir, data_set, run_config, model_config


def run_postprocess(all_results, log_dir):
    df = pd.DataFrame(all_results)
    mean_result = df.mean().to_dict()
    # 处理 auc：std 字段名为 "p"
    if len(df) > 1 and "auc" in df.columns:
        mean_result["p"] = df["auc"].std()
    else:
        mean_result["p"] = -1
    for metric in ["acc", "rmse"]:
        mean_result[f"p_{metric}"] = (
            df[metric].std() if len(df) > 1 and metric in df.columns else -1
        )

    pd.DataFrame([mean_result]).to_csv(
        os.path.join(log_dir, "test_result.csv"),
        index=False, encoding='utf-8', float_format='%.5f'
    )
    return mean_result


def main(model_name, data_name, data_base=None, logs_base=None, max_len=-1, folds=-1,  **kwargs):
    folds_params, log_dir, data_set, run_config, model_config = run_preprocess(model_name, data_name, data_base,
                                                                               logs_base, max_len, folds, **kwargs)
    try:
        all_results = []
        for datas_dir, logs_dir in folds_params:
            model = get_model(model_name, data_name, model_config)
            test_logs = train_loop(model, data_set, datas_dir, logs_dir, max_len, is_cross_validate=folds > 0, **kwargs)
            all_results.append(test_logs)
        final_result = run_postprocess(all_results, log_dir)
        print(final_result)
    except RuntimeError:
        traceback.print_exc()
        shutil.rmtree(log_dir, ignore_errors=True)
        return None

def default_args(parser):
    parser.add_argument('--logs_base', default='./logs', help='logs dir path')
    parser.add_argument('--data_base', default='./data', help='logs dir path')
    # parser.add_argument('--emb_type',type=str, default='', choices=['sep', 'opp','cross','pre'])#要结合对应模型修改
    parser.add_argument('--folds', type=int, default=-1, help='cross evaluate folds')
    parser.add_argument('--max_len', type=int, default=100, help='The max length of sequence')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda', 'npu'])
    parser.add_argument('--device_id', type=int, default='0')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size')
    parser.add_argument('--max_epochs', type=int, default=100, help='Max number of epochs for training')  ## 500
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'])
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--weight_decay', type=float, default=0, help='l2')
    parser.add_argument('--valid_interval', type=int, default=1, help='the number of epoch to eval')
    parser.add_argument('--patience', type=int, default=5, help='the number of epoch to wait before early stop')
    return parser
if __name__ == "__main__":
    main("DKT", "assist0910", "../data", "../logs", max_len=100, max_epochs=100, folds=-1)
