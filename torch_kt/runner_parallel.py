#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 2024/7/19 13:47
# @Author: hb925
# @File: runner.py
import copy
import datetime
import importlib
import inspect
import json
import os
import platform
import random
import shutil
import sys
import time
import traceback
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from torch_kt.runner import get_model, get_model_default_params, to_device, logs2str, run_preprocess, run_postprocess
from torch_kt.data_loader import DirDataset
from torch_kt.utils import root_dir
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


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
        model.module.reset_state()
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

    def metric_result(self, all_loss_global, all_preds_global, all_labels_global):
        metrics_dict = {}
        all_loss_global = all_loss_global.cpu().item()
        metrics_dict.update({"loss": all_loss_global})
        ts = all_labels_global.cpu().numpy()
        ps = all_preds_global.cpu().numpy()
        prelabels = [1 if p >= 0.5 else 0 for p in ps]
        auc = metrics.roc_auc_score(ts, ps)
        acc = metrics.accuracy_score(ts, prelabels)
        metrics_dict.update({"auc": auc, "acc": acc})
        return metrics_dict

    def collect_metrics(self, model, rank, world_size):
        """跨所有GPU收集 logits 和 labels 并计算全局指标"""
        # 将当前GPU的预测和标签转换为张量
        try:
            all_preds_tensor = torch.cat(model.module.all_preds)
            all_labels_tensor = torch.cat(model.module.all_labels)
            all_loss_global = torch.stack(model.module.all_losses).mean()
            torch.distributed.all_reduce(all_loss_global, op=torch.distributed.ReduceOp.SUM)
            all_loss_global = all_loss_global / world_size
            # 1. 计算所有GPU的长度
            preds_lengths = [torch.tensor([len(all_preds_tensor)], device=self.device) for _ in range(world_size)]
            dist.all_gather(preds_lengths, preds_lengths[0])
            preds_lengths = [int(l) for l in preds_lengths]

            # 2. 为每个GPU的预测结果填充到最大长度
            max_length = max(preds_lengths)
            padded_preds = torch.nn.functional.pad(all_preds_tensor, (0, max_length - len(all_preds_tensor)))
            padded_labels = torch.nn.functional.pad(all_labels_tensor, (0, max_length - len(all_labels_tensor)))
            # 3. 使用all_gather收集所有GPU的填充后的数据
            gathered_preds = [torch.zeros(max_length, device=self.device) for _ in range(world_size)]
            gathered_labels = [torch.zeros(max_length, device=self.device) for _ in range(world_size)]
            dist.all_gather(gathered_preds, padded_preds)
            dist.all_gather(gathered_labels, padded_labels)
            return gathered_preds, preds_lengths, gathered_labels, all_loss_global
        except:
            print(f"\n{rank}--collect_error")

    def fit(self, model, train_data, test_data, val_data=None, max_len=-1, local_rank=-1, rank=-1, world_size=-1,
            clean_weights=False,
            **kwargs):
        if len(kwargs) > 0:
            print(f"unused params for train:{kwargs}")
        # self.device = torch.device(f'{self.device}:{local_rank}')
        model.to(self.device)
        is_cpu = self.device == "cpu" if isinstance(self.device, str) else self.device.type == "cpu"
        # 将模型包装为DDP
        if not is_cpu:
            model = DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank
            )
        else:
            model = DistributedDataParallel(model)
        self.rest_state(model, training=True)
        self.on_train_begin()
        if rank == 0:
            os.makedirs(os.path.join(self.log_dir, "weights"), exist_ok=True)
        weight_path = os.path.join(self.log_dir, "weights", 'model.pth')
        csv_log_path = os.path.join(self.log_dir, "train.csv")
        test_log_path = os.path.join(self.log_dir, "test_result.csv")
        train_logs_data = []
        if val_data is None:
            val_data = test_data
        total_batch = None
        progressbar_str = ""
        max_auc = 0
        best_epoch = -1
        width = 30

        for epoch in range(self.max_epochs):
            model.train()
            model.module.reset_state()
            train_data.sampler.set_epoch(epoch)
            self.on_epoch_begin(epoch)
            if rank == 0:
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
                tmp_logs = model.module.train_step(data)
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
                if rank == 0:
                    print(len(last_train_out_str) * "\b" + "\r" + train_out_str_add, end="")
                last_train_out_str = train_out_str_add
                self.on_train_batch_end(batch, logs=logs)
            if total_batch is None:
                total_batch = batch + 1
            gathered_preds, preds_lengths, gathered_labels, all_loss_global = self.collect_metrics(model, rank,
                                                                                                   world_size)
            # 去掉填充部分
            all_preds_global = []
            all_labels_global = []
            for i in range(world_size):
                all_preds_global.append(gathered_preds[i][:preds_lengths[i]])
                all_labels_global.append(gathered_labels[i][:preds_lengths[i]])
            all_preds_global = torch.cat(all_preds_global)
            all_labels_global = torch.cat(all_labels_global)
            tr_logs = self.metric_result(all_loss_global, all_preds_global, all_labels_global)
            tr_logs = {
                "train_" + name: val for name, val in tr_logs.items()
            }
            trlogs_str = logs2str(tr_logs)
            if rank == 0:
                print(len(last_train_out_str) * "\b" + "\r" + progressbar_str + trlogs_str, end="")
            if epoch % self.valid_interval == 0:
                dist.barrier()
                val_data.sampler.set_epoch(epoch)
                val_logs = self.evaluate(model, val_data, rank, world_size)
                if "auc" in val_logs.keys() and val_logs["auc"] - max_auc > 0.0001:
                    max_auc = val_logs["auc"]
                    best_epoch = epoch
                    if rank == 0:
                        model.to("cpu")
                        torch.save(model.state_dict(), weight_path,_use_new_zipfile_serialization=True)
                        model.to(self.device)
                val_logs = {
                    "val_" + name: val for name, val in val_logs.items()
                }
                dist.barrier()
            if rank == 0:
                print("\r" + progressbar_str + trlogs_str + "," + logs2str(val_logs))
            train_log_record = {"enpoch": epoch}
            train_log_record.update(tr_logs)
            train_log_record.update(val_logs)
            train_logs_data.append(train_log_record)
            enpoch_logs = copy.copy(tr_logs)
            enpoch_logs.update(val_logs)
            self.on_epoch_end(epoch, logs=enpoch_logs)
            dist.barrier()
            if self._should_stop(epoch, best_epoch):
                break
        self.on_train_end()
        dist.barrier()
        stat_dict = None
        if rank == 0:
            pd.DataFrame(train_logs_data).to_csv(csv_log_path, index=False, encoding='utf-8',
                                                 float_format='%.5f')
            if os.path.exists(weight_path):
                stat_dict = torch.load(weight_path, map_location='cpu', weights_only=True)
        #         model.load_state_dict(stat_dict)
        #         # 广播到所有GPU
        #         for param in model.parameters():
        #             dist.broadcast(param.data, src=0)
        # else:
        #     if os.path.exists(weight_path):
        #         # 接收广播
        #         for param in model.parameters():
        #             dist.broadcast(param.data, src=0)
        dist.broadcast_object_list([stat_dict], src=0)  # 所有 rank 都调用

        if stat_dict is not None:
            model.load_state_dict(stat_dict)
        # 关键同步点：确保所有GPU都完成了参数加载
        dist.barrier()  # 这是必要的，因为后续训练/验证依赖于模型参数
        test_logs = self.evaluate(model, test_data, rank, world_size)
        if rank == 0:
            self.on_test_end(logs=test_logs)
            test_logs1 = {
                "test_" + name: val for name, val in test_logs.items()
            }
            print(logs2str(test_logs1))
            pd.DataFrame([test_logs]).to_csv(test_log_path, index=False, encoding='utf-8',
                                             float_format='%.5f')
            if clean_weights:
                shutil.rmtree(os.path.join(self.log_dir, "weights"))
        return test_logs

    def evaluate(self, model, data_set, rank, world_size):
        self.rest_state(model, training=False)
        with torch.no_grad():
            self.on_test_begin()
            for step, data in enumerate(data_set):
                data = to_device(data, self.device)
                model.module.test_step(data)
        dist.barrier()
        gathered_preds, preds_lengths, gathered_labels, all_loss_global = self.collect_metrics(model, rank, world_size)
        # 去掉填充部分
        all_preds_global = []
        all_labels_global = []
        for i in range(world_size):
            all_preds_global.append(gathered_preds[i][:preds_lengths[i]])
            all_labels_global.append(gathered_labels[i][:preds_lengths[i]])
        all_preds_global = torch.cat(all_preds_global)
        all_labels_global = torch.cat(all_labels_global)
        tr_logs = self.metric_result(all_loss_global, all_preds_global, all_labels_global)
        self.on_test_end()
        return tr_logs


def train_loop(model, dataname, datas_dir, log_dir, max_len, batch_size=32, max_epochs=50, lr=0.01, optimizer="adam",
               patience=5, valid_interval=1, device="cuda", rank=-1, world_size=-1, local_rank=-1, weight_decay=0,
               is_cross_validate=False, fast_run=False, **kwargs):
    device, _ = setup_device(local_rank, device)
    feature_names, label_names = model.inputs_specs
    trdata_loader, vadata_loader, tedata_loader = dataname.split_generator(datas_dir, feature_names,
                                                                           label_names,
                                                                           sample_num=256 if fast_run else -1,
                                                                           max_len=max_len)
    train_sampler = DistributedSampler(trdata_loader, num_replicas=world_size,
                                       rank=rank,
                                       shuffle=True,
                                       drop_last=True)  # 确保每个GPU样本数相同
    valid_sampler = DistributedSampler(vadata_loader, num_replicas=world_size,
                                       rank=rank,
                                       shuffle=False,
                                       drop_last=True)  # 确保每个GPU样本数相同
    train_dataloader = DataLoader(trdata_loader, batch_size=batch_size, sampler=train_sampler)
    val_dataloader = DataLoader(vadata_loader, batch_size=batch_size, sampler=valid_sampler)
    if is_cross_validate:
        test_dataloader = val_dataloader
    else:
        test_sampler = DistributedSampler(tedata_loader, num_replicas=world_size,
                                          rank=rank,
                                          shuffle=False,
                                          drop_last=True)  # 确保每个GPU样本数相同
        test_dataloader = DataLoader(tedata_loader, batch_size=batch_size, sampler=test_sampler)
    model.compile_model(optimizer=optimizer, lr=lr, weight_decay=weight_decay)
    trainner = KtTrainner(log_dir, max_epochs,
                          device=device, patience=patience, valid_interval=valid_interval)
    test_logs = trainner.fit(model,
                             train_dataloader,
                             test_dataloader,
                             val_dataloader, max_len, local_rank, rank, world_size,
                             **kwargs)
    return test_logs


def get_distributed_backend():
    """
    根据操作系统自动选择合适的分布式后端
    """
    system = platform.system().lower()

    if system == 'linux':
        # Linux: 优先使用 nccl（GPU）
        if torch.cuda.is_available():
            return 'nccl'
        elif hasattr(torch, 'npu') and torch.npu.is_available():
            return 'hccl'
        else:
            return 'gloo'  # 如果没有 GPU，Linux 也可用 gloo 或 mpi
    elif system == 'windows':
        # Windows: 只能使用 gloo（或 mpi），nccl 不支持
        return 'gloo'
    elif system == 'darwin':  # macOS
        return 'gloo'  # macOS 不支持 nccl，通常用 gloo
    else:
        raise RuntimeError(f"Unsupported operating system: {system}")


def setup_device(local_rank, params_device):
    is_windows = os.name == 'nt'  # Windows
    has_cuda = torch.cuda.is_available() and params_device == "cuda"
    has_npu = hasattr(torch, 'npu') and torch.npu.is_available() and params_device == "npu"
    # 3. 设置设备和 backend
    if has_npu:
        # 华为 NPU（通常只在 Linux 上）
        torch.npu.set_device(local_rank)
        device = torch.device(f'npu:{local_rank}')
        backend = 'hccl'  # 华为集合通信库

    elif has_cuda and not is_windows:
        # NVIDIA GPU + 非 Windows（Linux/macOS）
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        backend = 'nccl'  # 高性能 GPU 通信

    elif has_cuda and is_windows:
        # Windows 上 CUDA 只能用 gloo
        device = torch.device(f'cuda:{local_rank}')
        backend = 'gloo'
        # 注意：Windows 上 set_device 对 cuda 无效，但 to('cuda:x') 仍可用

    else:
        # CPU fallback
        device = torch.device('cpu')
        backend = 'gloo'  # CPU 通信
    return device, backend


def setup_distributed():
    """
    跨平台、跨硬件的分布式设置
    返回: device, backend, rank, world_size
    """
    # 1. 获取 local_rank（来自 torchrun 或手动设置）
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    # 2. 检测可用硬件和操作系统
    is_windows = os.name == 'nt'  # Windows
    has_cuda = torch.cuda.is_available()
    has_npu = hasattr(torch, 'npu') and torch.npu.is_available()

    # 3. 设置设备和 backend
    if has_npu:
        # 华为 NPU（通常只在 Linux 上）
        torch.npu.set_device(local_rank)
        device = torch.device(f'npu:{local_rank}')
        backend = 'hccl'  # 华为集合通信库

    elif has_cuda and not is_windows:
        # NVIDIA GPU + 非 Windows（Linux/macOS）
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        backend = 'nccl'  # 高性能 GPU 通信

    elif has_cuda and is_windows:
        # Windows 上 CUDA 只能用 gloo
        device = torch.device(f'cuda:{local_rank}')
        backend = 'gloo'
        # 注意：Windows 上 set_device 对 cuda 无效，但 to('cuda:x') 仍可用

    else:
        # CPU fallback
        device = torch.device('cpu')
        backend = 'gloo'  # CPU 通信

    # 4. 仅在多进程时初始化分布式
    if world_size > 1:
        # 设置 master 信息（torchrun 会自动设置，这里为 spawn 兼容）
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = '127.0.0.1'
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '12355'

        # 初始化进程组
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=60)
        )

        print(f"[Rank {rank}] Using device={device}, backend={backend}, world_size={world_size}")

    return device, backend, rank, world_size


def main(model_name, data_name, data_base=None, logs_base=None, max_len=-1, folds=-1, rank=-1, world_size=-1, **kwargs):
    folds_params, log_dir, data_set, run_config, model_config = run_preprocess(model_name, data_name, data_base,
                                                                               logs_base, max_len, folds, **kwargs)
    try:
        if kwargs["device"] == "npu":
            import torch_npu
        backend = get_distributed_backend()
        dist.init_process_group(backend=backend, rank=rank,
                                world_size=world_size,
                                timeout=datetime.timedelta(seconds=60))
        # dist.barrier()
        all_results = []
        for datas_dir, logs_dir in folds_params:
            model = get_model(model_name, data_name, model_config)
            test_logs = train_loop(model, data_set, datas_dir, logs_dir, max_len, is_cross_validate=folds > 0,
                                   rank=rank, world_size=world_size, **kwargs)
            all_results.append(test_logs)
        if rank == 0:
            final_result = run_postprocess(all_results, log_dir)
            print(final_result)
    except RuntimeError:
        traceback.print_exc()
        shutil.rmtree(log_dir, ignore_errors=True)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()  # 先同步所有 rank
            dist.destroy_process_group()
        print(f"Rank {rank} cleaned up.")


def main_wrapper(local_rank, args):
    args.local_rank = local_rank
    args.rank = local_rank
    # if args.rank == 0:
    #     os.environ['MASTER_ADDR'] = 'localhost'
    #     os.environ['MASTER_PORT'] = '12355'
    main(**vars(args))


if __name__ == "__main__":
    main("DKT", "assist0910", "../data", "../logs", max_len=100, max_epochs=100, folds=-1)
