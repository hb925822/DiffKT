#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DiffKT — Stage 1: Global Context-aware State Estimator (GCSE).

Bidirectional LSTM estimator used to build *process-consistent* latent state
targets from the full interaction sequence. The checkpoint produced by this
script is consumed by ``diff_prelstm_as2.py`` (Stage 2), where every parameter
whose name matches a key in the checkpoint is frozen.

Key point regarding the paper's data-leakage discussion: full-sequence access
happens **only here**, during offline training. The frozen estimator is never
invoked at inference time, so no future response can influence a prediction.

Usage
-----
    # Train on the 80% training/validation pool (default)
    python bikt_pres.py --data_name assist0910 --data_base ../data \\
        --logs_base ../logs --emb_size 128 --lr 1e-3 --save

    # Train on all data (used to produce the released GCSE checkpoint)
    python bikt_pres.py --data_name assist0910 --save --all_data

Note: run this script from the ``code/`` directory (or add it to PYTHONPATH)
so that the sibling ``torch_kt`` package can be imported.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch.nn import Embedding, LSTM, Dropout
from torch.utils.data import ConcatDataset, DataLoader

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from torch_kt.data_loader import DirDataset  # noqa: E402
from torch_kt.runner import logs2str, to_device  # noqa: E402

__all__ = ["BiktPreS", "save_pretrain_model", "main"]


class BiktPreS(torch.nn.Module):
    """Bidirectional LSTM state estimator (GCSE in the paper).

    The module produces three latent states per time step:

    ``pad_h``      forward-only (causal) estimate
    ``pad_h_rev``  backward-only (anti-causal) estimate
    ``h_con``      their sum — the full-sequence, process-consistent state
                   used as the diffusion target ``h_t^0`` in Stage 2.

    Prediction follows the cosine-similarity read-out of the original
    bidirectional knowledge-tracing formulation, rescaled from [-1, 1] to [0, 1].
    """

    def __init__(self, skill_num: int, emb_size: int = 128, dropout: float = 0.1,
                 use_smooth: bool = False, **kwargs) -> None:
        # ``use_smoth`` was the historical (misspelled) keyword; accept it so that
        # configs written by older runs still load.
        if "use_smoth" in kwargs and "use_smooth" not in kwargs:
            use_smooth = kwargs.pop("use_smoth")
        if len(kwargs) > 0:
            print(f"unused params for model:{kwargs}")
        super().__init__()
        self.name = "BiktPreS"
        self._losses: List[float] = []
        self._labels: List[np.ndarray] = []
        self._outputs: List[np.ndarray] = []
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.num_c = skill_num
        self.emb_size = emb_size
        self.hidden_units = emb_size
        self.dropout = dropout
        self.use_smooth = use_smooth

        self.skill_emb = Embedding(self.num_c, self.emb_size)
        self.lstm_layer = LSTM(self.emb_size, self.hidden_units, batch_first=True)
        self.lstm_layer_reverse = LSTM(self.emb_size, self.hidden_units, batch_first=True)
        self.dropout_layer = Dropout(self.dropout)
        self.layer_normal = torch.nn.LayerNorm(self.hidden_units)

    def compute_hide_state(
        self, qa_emb: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run both LSTM directions and align their outputs into one sequence."""
        mask_rev = mask.type(torch.float32).flip(dims=[1])
        qa_emb_rev = qa_emb.flip(dims=[1])
        qa_emb_rev = qa_emb_rev * mask_rev

        h, _ = self.lstm_layer(qa_emb)
        h_rev, _ = self.lstm_layer_reverse(qa_emb_rev)
        h_rev = h_rev.flip(dims=[1])

        pad = torch.zeros([h.shape[0], 1, h.shape[-1]], device=h.device)
        pad_h = torch.cat([pad, h[:, :-1, :]], dim=1)
        pad_h_rev = torch.cat([h_rev[:, 1:, :], pad], dim=1)
        h_con = pad_h + pad_h_rev
        return h_con, pad_h, pad_h_rev

    def forward(self, x, mask: Optional[torch.Tensor] = None, training: Optional[bool] = None, **kwargs):
        qa, q, r = x
        q_emb = self.skill_emb(q)
        res = r.unsqueeze(dim=-1)
        qa_emb = res * q_emb - (1 - res) * q_emb

        h_con, _, _ = self.compute_hide_state(qa_emb, mask)
        h_con = self.dropout_layer(h_con)

        # Cosine similarity between the concept embedding and the state,
        # mapped to a probability in [0, 1].
        y = torch.sum(q_emb * h_con, keepdim=True, dim=-1)
        y = y / (torch.norm(q_emb, p=2, dim=-1, keepdim=True)
                 * torch.norm(h_con, p=2, dim=-1, keepdim=True))
        y = y / 2 + 0.5

        y = y[:, 1:, :]
        return y, None

    # ------------------------------------------------------------------
    # Optimisation / bookkeeping
    # ------------------------------------------------------------------
    def compile_model(self, optimizer: str = None, lr: float = 0.001, weight_decay: float = 0) -> None:
        if str.lower(optimizer) == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif str.lower(optimizer) == "sgd":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError("unknow optimizer name")

    def reset_state(self) -> None:
        self._losses.clear()
        self._labels.clear()
        self._outputs.clear()

    def compute_loss(self, x, y_pred, y, sample_weight=None) -> torch.Tensor:
        return F.binary_cross_entropy(y_pred, y)

    def compute_metrics(self) -> Dict[str, float]:
        loss = np.mean(self._losses)
        ts = np.concatenate(self._labels, axis=0)
        ps = np.concatenate(self._outputs, axis=0)
        prelabels = [1 if p >= 0.5 else 0 for p in ps]
        auc = metrics.roc_auc_score(ts, ps)
        acc = metrics.accuracy_score(ts, prelabels)
        return {"loss": loss, "auc": auc, "acc": acc}

    def train_step(self, data) -> Dict[str, float]:
        x, y, mask, sample_weight = self.data_map(data)
        y_pred, _smooth_loss = self(x, training=True, mask=mask)
        mask = mask[:, 1:, :]
        y_pred, y = y_pred.masked_select(mask), y.masked_select(mask)
        loss = self.compute_loss(x, y_pred, y, sample_weight)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        y_pred_np, y_np = y_pred.detach().cpu().numpy(), y.detach().cpu().numpy()
        loss_v = loss.detach().cpu().item()
        self._labels.append(y_np)
        self._outputs.append(y_pred_np)
        self._losses.append(loss_v)
        return {"loss": loss_v}

    def test_step(self, data) -> Dict[str, float]:
        x, y, mask, sample_weight = self.data_map(data)
        y_pred, _ = self(x, training=False, mask=mask)
        mask = mask[:, 1:, :]
        y_pred, y = y_pred.masked_select(mask), y.masked_select(mask)
        loss = self.compute_loss(x, y_pred, y, sample_weight)

        y_pred_np, y_np = y_pred.detach().cpu().numpy(), y.detach().cpu().numpy()
        loss_v = loss.detach().cpu().item()
        self._labels.append(y_np)
        self._outputs.append(y_pred_np)
        self._losses.append(loss_v)
        return {"loss": loss_v}

    # ------------------------------------------------------------------
    # Training driver
    # ------------------------------------------------------------------
    def fit(self, train_data, test_data, val_data=None, log_dir: str = None, max_epochs: int = 1,
            device: str = "cpu", patience: int = 10, valid_interval: int = 1, **kwargs) -> Dict[str, float]:
        if len(kwargs) > 0:
            print(f"unused params for train:{kwargs}")
        self.to(device)
        self.reset_state()

        os.makedirs(os.path.join(log_dir, "weights"), exist_ok=True)
        weight_path = os.path.join(log_dir, "weights", "model.pth")
        csv_log_path = os.path.join(log_dir, "train.csv")
        test_log_path = os.path.join(log_dir, "test_result.csv")

        train_logs_data = []
        if val_data is None:
            val_data = test_data
        total_batch = None
        progressbar_str = ""
        max_auc = 0
        best_epoch = -1
        width = 30

        for epoch in range(max_epochs):
            self.train()
            print(f"Epoch {epoch + 1}/{max_epochs}")
            last_train_out_str = ""
            epoch_start = time.time()
            batch = 0
            logs = {}
            val_logs = {}

            for batch, data in enumerate(train_data):
                step_start = time.time()
                data = to_device(data, device)
                logs = self.train_step(data)

                now = time.time()
                time_step = round((now - step_start) * 1000)
                progressbar_str = ""
                if total_batch is None:
                    progressbar_str += f"{batch + 1:7d}/unknown,[{'=' * width}]"
                else:
                    numdigits = int(np.log10(total_batch)) + 1
                    prog_width = round((batch + 1) / total_batch * width)
                    progressbar_str += (
                        f"{batch + 1:{numdigits}d}/{total_batch},"
                        f"[{'=' * prog_width}{'.' * round(width - prog_width)}]"
                    )
                if total_batch is not None and batch + 1 == total_batch:
                    progressbar_str += f"- {round(now - epoch_start):^3d}s :{time_step}ms/step,"
                else:
                    progressbar_str += f"- ETA :{time_step}ms/step,"

                train_out_str_add = progressbar_str + logs2str(logs)
                print(len(last_train_out_str) * "\b" + "\r" + train_out_str_add, end="")
                last_train_out_str = train_out_str_add

            tr_logs = self.compute_metrics()
            tr_logs = {"train_" + name: val for name, val in tr_logs.items()}
            trlogs_str = logs2str(tr_logs)
            print(len(last_train_out_str) * "\b" + "\r" + progressbar_str + trlogs_str, end="")

            if total_batch is None:
                total_batch = batch + 1

            if epoch % valid_interval == 0:
                val_logs = self.evaluate(val_data, device=device)
                if val_logs["auc"] - max_auc > 0.0001:
                    max_auc = val_logs["auc"]
                    best_epoch = epoch
                    torch.save(self.state_dict(), weight_path)
                    torch.save(
                        self.state_dict(),
                        os.path.join(log_dir, "weights", f"model_{epoch:03d}_{round(max_auc * 1000):04d}.pth"),
                    )
                val_logs = {"val_" + name: val for name, val in val_logs.items()}

            print("\r" + progressbar_str + trlogs_str + "," + logs2str(val_logs))
            train_log_record = {"enpoch": epoch}
            train_log_record.update(tr_logs)
            train_log_record.update(val_logs)
            train_logs_data.append(train_log_record)

            if 0 < patience * valid_interval < (epoch - best_epoch):
                break

        pd.DataFrame(train_logs_data).to_csv(csv_log_path, index=False, encoding="utf-8", float_format="%.5f")

        if os.path.exists(weight_path):
            self.load_state_dict(torch.load(weight_path))

        test_logs = self.evaluate(test_data, device=device)
        test_logs1 = {"test_" + name: val for name, val in test_logs.items()}
        print(logs2str(test_logs1))
        pd.DataFrame([test_logs]).to_csv(test_log_path, index=False, encoding="utf-8", float_format="%.5f")
        return test_logs

    def evaluate(self, data_set, device: str = "cpu") -> Dict[str, float]:
        self.to(device)
        self.reset_state()
        self.eval()
        with torch.no_grad():
            for _step, data in enumerate(data_set):
                data = to_device(data, device)
                self.test_step(data)
        return self.compute_metrics()

    def predict(self, data_set, device: str = "cpu", log_dir: str = None) -> None:
        """Export the combined, forward-only and backward-only predictions."""
        self.to(device)
        self.reset_state()
        self.eval()
        predict_path = os.path.join(log_dir, "predicts")
        os.makedirs(predict_path, exist_ok=True)
        with torch.no_grad():
            for step, data in enumerate(data_set):
                data = to_device(data, device)
                x, y, mask, sample_weight = self.data_map(data)
                _qa, q, r = x
                q_emb = self.skill_emb(q)
                res = r.unsqueeze(dim=-1)
                qa_emb = res * q_emb - (1 - res) * q_emb
                h_con, pad_h, pad_h_rev = self.compute_hide_state(qa_emb, mask)

                y_pred = torch.sigmoid(torch.sum(q_emb * h_con, keepdim=True, dim=-1))[:, 1:, :]
                y_rev = torch.sigmoid(torch.sum(q_emb * pad_h_rev, keepdim=True, dim=-1))[:, 1:, :]
                y_pre = torch.sigmoid(torch.sum(q_emb * pad_h, keepdim=True, dim=-1))[:, 1:, :]

                mask_flat = mask[:, 1:, :].reshape(-1).detach().cpu().numpy()
                pad = (mask_flat - 1) * np.ones_like(mask_flat)

                def flat(t):
                    v = t.reshape(-1).detach().cpu().numpy()
                    return (v * mask_flat + pad).tolist()

                predict_datas = {
                    "skill": flat(q[:, 1:]),
                    "response": y.reshape(-1).detach().cpu().numpy().tolist(),
                    "predict": flat(y_pred),
                    "predict_pre": flat(y_pre),
                    "predict_rev": flat(y_rev),
                }
                pd.DataFrame(predict_datas).to_csv(
                    os.path.join(predict_path, f"batch_{step}.csv"),
                    index=False, encoding="utf-8", float_format="%.3f",
                )

    # ------------------------------------------------------------------
    # Data plumbing
    # ------------------------------------------------------------------
    @property
    def inputs_specs(self):
        return ("skill_response", "skill"), "correct"

    def data_map(self, data):
        (skill_response, skill), y = data
        mask = torch.ge(y, 0).type(torch.int8)
        skill_response = (skill_response * mask).long()
        skill = (skill * mask).long()
        r = (y * mask).long()
        y = y.unsqueeze(-1).type(torch.float)
        return (skill_response, skill, r), y[:, 1:], mask.unsqueeze(-1).type(torch.bool), None


# ---------------------------------------------------------------------------
# Checkpoint export
# ---------------------------------------------------------------------------
def save_pretrain_model(pretrain_log_dir: str, premodel_name: str, dname: str) -> None:
    """Copy the trained weights into ``./pretrain_model/`` for Stage 2.

    The file name encodes the dataset, whether all data was used, and the
    weight-decay value, so that Stage 2 can select the right checkpoint via
    ``--model_path``.
    """
    pretrain_log_dir = os.path.abspath(pretrain_log_dir)
    if not os.path.isdir(pretrain_log_dir):
        return
    test_result_file = os.path.join(pretrain_log_dir, "test_result.csv")
    config_file0 = os.path.join(pretrain_log_dir, "configs/config.json")
    config_file1 = os.path.join(pretrain_log_dir, "run_config.json")
    config_file = config_file0 if os.path.exists(config_file0) else config_file1

    if not (os.path.exists(test_result_file) and os.path.exists(config_file)):
        return

    with open(config_file, "r") as f:
        config = json.load(f)

    new_pth_name = f"{premodel_name}_{dname}"
    if config.get("all_data"):
        new_pth_name += "_all"
    if config.get("weight_decay", 0) > 1e-10:
        new_pth_name += "_" + str(config["weight_decay"]).replace(".", "#")

    out_dir = os.path.join(os.getcwd(), "pretrain_model")
    os.makedirs(out_dir, exist_ok=True)
    weights_dir = os.path.join(pretrain_log_dir, "weights")
    if not os.path.isdir(weights_dir):
        return
    for weight_file in os.listdir(weights_dir):
        target_file = os.path.join(out_dir, f"{new_pth_name}{weight_file.replace('model', '')}")
        print(target_file)
        shutil.copyfile(os.path.join(weights_dir, weight_file), target_file)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Stage-1 GCSE estimator.")
    parser.add_argument("--data_name", default="assist0910",
                        help="Dataset name, e.g. assist0910 / algebra2005 / bridge2algebra2006 / nips")
    parser.add_argument("--logs_base", default="../logs", help="Directory for logs and checkpoints")
    parser.add_argument("--data_base", default="../data", help="Directory containing the preprocessed datasets")
    parser.add_argument("--max_len", type=int, default=100, help="Maximum interaction sequence length")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--emb_size", default=128, type=int, help="Embedding size")
    parser.add_argument("--hidden_units", default=128, type=int, help="Hidden size")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of training epochs")
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["SGD", "Adam"])
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay")
    parser.add_argument("--use_smooth", action="store_true", default=False,
                        help="Ablation: add the state-smoothness auxiliary loss")
    parser.add_argument("--valid_interval", type=int, default=1, help="Validate every N epochs")
    parser.add_argument("--patience", type=int, default=5, help="Early-stopping patience (epochs)")
    parser.add_argument("--all_data", action="store_true", default=False,
                        help="Also train on the validation and test partitions")
    parser.add_argument("--save", action="store_true", default=False,
                        help="Copy the trained weights into ./pretrain_model/")
    return parser


def main(argv=None) -> Dict[str, float]:
    args = build_argparser().parse_args(argv)

    if not torch.cuda.is_available():
        if args.device == "cuda":
            print("warning:cuda is not enable!!!!!!!!!!!!!!!!!!")
        args.device = "cpu"

    datas_dir = os.path.join(os.path.abspath(args.data_base), args.data_name)
    dataset = DirDataset(datas_dir)
    dataset.recheck_data(max_len=args.max_len)

    model_configs = vars(args)
    model_configs.update(dataset.params)
    model = BiktPreS(**model_configs)
    model.compile_model(optimizer=args.optimizer, lr=args.lr, weight_decay=args.weight_decay)

    log_path = f'{args.data_name}_{model.name}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'
    log_dir = os.path.join(os.path.abspath(args.logs_base), log_path)
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(model_configs, f, indent=4)

    feature_names, label_names = model.inputs_specs
    train_data, val_data, test_data = dataset.split_generator(
        datas_dir, feature_names, label_names, max_len=args.max_len
    )
    if args.all_data:
        train_data = ConcatDataset([train_data, val_data, test_data])

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size)

    test_logs = model.fit(train_dataloader, test_dataloader, val_dataloader, log_dir, **model_configs)
    model.predict(test_dataloader, args.device, log_dir)

    if args.save:
        save_pretrain_model(log_dir, model.name, args.data_name)
    return test_logs


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    main()
