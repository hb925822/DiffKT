#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 2026/1/28 23:12
# @Author: hb925
# @File: efficiency_monitor.py
import torch
import time
import json
import os
from typing import Optional, Dict, Any, List


class TrainingEfficiencyMonitor:
    def __init__(
        self,
        model: torch.nn.Module,
        per_epoch: bool = True,
        json_path: Optional[str] = None,
        append: bool = True
    ):
        """
        初始化监控器。

        Args:
            model (torch.nn.Module): 要监控的模型。
            per_epoch (bool):
                - True: 每个 epoch 单独记录和重置指标（推荐）。
                - False: 全局累计记录。
            json_path (str or None):
                - 若提供，会将每个 epoch 的 summary 写入该路径。
                - 推荐使用 .jsonl（JSON Lines）格式以支持流式写入。
            append (bool):
                - True: 追加写入（适合 per_epoch=True）。
                - False: 覆盖写入（需手动控制）。
        """
        self.model = model
        self.per_epoch = per_epoch
        self.json_path = json_path
        self.append = append
        self.total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # 存储所有 epoch 的结果（用于最终 dump）
        self.all_summaries: List[Dict[str, Any]] = []

        # 时间相关
        self.epoch_start_time: Optional[float] = None
        self.train_step_times = []
        self.infer_step_times = []

        # GPU 内存
        self.device = next(model.parameters()).device
        self._cuda_available = self.device.type == 'cuda'

        self.reset()

    def reset(self):
        """重置所有可变指标（用于每个 epoch 开始时调用）"""
        self.epoch_start_time = time.time()
        self.train_step_times.clear()
        self.infer_step_times.clear()
        if self._cuda_available:
            torch.cuda.reset_peak_memory_stats(self.device)

    def start_train_step(self):
        self._train_step_start = time.time()

    def end_train_step(self):
        step_time = time.time() - self._train_step_start
        self.train_step_times.append(step_time)

    def start_infer_step(self):
        self._infer_step_start = time.time()

    def end_infer_step(self):
        step_time = time.time() - self._infer_step_start
        self.infer_step_times.append(step_time)

    def get_current_gpu_memory_mb(self) -> float:
        if not self._cuda_available:
            return 0.0
        return torch.cuda.memory_allocated(self.device) / (1024 ** 2)

    def get_peak_gpu_memory_mb(self) -> float:
        if not self._cuda_available:
            return 0.0
        return torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)

    def get_epoch_time(self) -> float:
        if self.epoch_start_time is None:
            return 0.0
        return time.time() - self.epoch_start_time

    def get_summary(self, epoch: Optional[int] = None) -> Dict[str, Any]:
        """获取当前指标摘要，可选包含 epoch 编号"""
        avg_train_step = sum(self.train_step_times) / len(self.train_step_times) if self.train_step_times else 0.0
        avg_infer_step = sum(self.infer_step_times) / len(self.infer_step_times) if self.infer_step_times else 0.0

        summary = {
            "epoch": epoch,
            "total_trainable_params": self.total_params,
            "avg_train_step_time_sec": avg_train_step,
            "avg_inference_step_time_sec": avg_infer_step,
            "epoch_total_time_sec": self.get_epoch_time(),
            "peak_gpu_memory_mb": self.get_peak_gpu_memory_mb(),
            "current_gpu_memory_mb": self.get_current_gpu_memory_mb(),
        }
        return summary

    def print_summary(self, epoch: Optional[int] = None):
        s = self.get_summary(epoch=epoch)
        prefix = f"Epoch {epoch}: " if epoch is not None else ""
        print(f"\n{prefix}Training Efficiency Summary:")
        print(f"  Total Trainable Parameters: {s['total_trainable_params']:,}")
        print(f"  Avg Train Step Time:        {s['avg_train_step_time_sec']:.4f} sec")
        print(f"  Avg Inference Step Time:    {s['avg_inference_step_time_sec']:.4f} sec")
        print(f"  Epoch Total Time:           {s['epoch_total_time_sec']:.2f} sec")
        print(f"  Peak GPU Memory:            {s['peak_gpu_memory_mb']:.2f} MB")
        print(f"  Current GPU Memory:         {s['current_gpu_memory_mb']:.2f} MB")

    def save_to_json(self, epoch: Optional[int] = None):
        """
        将当前 summary 保存到 JSON。
        - 如果 json_path 是 .jsonl，每调用一次写一行（推荐）。
        - 如果是 .json，会累积并在训练结束时 dump（需调用 finalize_json）。
        """
        summary = self.get_summary(epoch=epoch)

        if self.json_path is None:
            return

        # 累积到内存（用于最终 dump）
        self.all_summaries.append(summary)

        # 自动推断格式
        _, ext = os.path.splitext(self.json_path)
        if ext == ".jsonl":
            # 流式写入：每次 append 一行
            mode = "a" if self.append else "w"
            with open(self.json_path, mode, encoding="utf-8") as f:
                f.write(json.dumps(summary, indent=None) + "\n")
        elif ext == ".json":
            # 暂不写入，等 finalize_json 时统一写（避免频繁 IO）
            pass
        else:
            # 默认当作 .jsonl 处理
            mode = "a" if self.append else "w"
            with open(self.json_path, mode, encoding="utf-8") as f:
                f.write(json.dumps(summary, indent=None) + "\n")

    def finalize_json(self):
        """
        （可选）如果使用 .json 格式，调用此方法将 all_summaries 一次性写入。
        对 .jsonl 无需调用。
        """
        if self.json_path is None:
            return
        _, ext = os.path.splitext(self.json_path)
        if ext == ".json":
            os.makedirs(os.path.dirname(self.json_path), exist_ok=True)
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(self.all_summaries, f, indent=2)