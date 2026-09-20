#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 2026/2/1 22:24
# @Author: hb925
# @File: run_for_predict.py
import argparse
import json
import os
import os.path
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from torch_kt.data_loader import DirDataset
from torch_kt.runner import get_model, get_model_default_params, to_device


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser(description='predict for model')
parser.add_argument('log_dir', nargs='?',default='logs', help='Model train log dir')
parser.add_argument('--result_base', default='./predict_result', help='predict result dir path')
parser.add_argument('--data_base', default='./data', help='logs dir path')
parser.add_argument('--predict_type', default='single', help='predict output')
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda', 'npu'])



def save_predictions_to_csv(save_path, all_preds, all_labels, save_index=True, expand_sequences=False):
    """
    将模型预测结果和真实标签保存为CSV文件。支持两种模式：
    1. 按样本保存 (默认): 每个样本一行，序列以逗号分隔的字符串形式存储。
    2. 完全展开保存: 将每个序列展开，每个时间步一行。

    参数:
    - save_path (str): CSV文件的保存路径，例如 'results/predictions.csv'。
    - all_preds (list of np.ndarray): 包含每个批次预测结果的列表。
                                      每个元素应为形状为 (batch_size, seq_len) 的数组。
    - all_labels (list of np.ndarray): 包含每个批次真实标签的列表。
                                       每个元素应为形状为 (batch_size, seq_len) 的数组。
    - save_index (bool, optional): 是否在CSV中保存样本索引列（仅在 expand_sequences=False 时有效）。
                                     默认为 True。
    - expand_sequences (bool, optional): 是否将序列完全展开。如果为True，则每个时间步占一行，
                                         并与对应的样本索引和时间步索引一起保存。默认为 False。
    """
    # --- 输入校验 ---
    if not all_preds or not all_labels:
        print("Warning: 'all_preds' or 'all_labels' is empty. Nothing to save.")
        return

    # --- 1. 合并所有批次的数据 ---
    try:
        all_preds_array = np.concatenate(all_preds, axis=0)
        all_labels_array = np.concatenate(all_labels, axis=0)
    except ValueError as e:
        print(f"Error concatenating arrays: {e}")
        print("Please ensure 'all_preds' and 'all_labels' are non-empty lists of "
              "numpy arrays with the same number of batches and compatible shapes.")
        return

    print(f"Successfully concatenated data. Shape of predictions: {all_preds_array.shape}")

    # --- 根据模式选择不同的处理逻辑 ---
    if not expand_sequences:
        # --- 模式 A: 按样本保存 (默认) ---
        print("Saving in 'per-sample' mode.")
        num_samples = all_preds_array.shape[0]
        csv_data = []
        for i in range(num_samples):
            row_data = {
                'true_label_sequence': ','.join(map(str, all_labels_array[i, :])),
                'predicted_sequence': ','.join(map(str, all_preds_array[i, :]))
            }
            if save_index:
                row_data['sample_index'] = i
            csv_data.append(row_data)

        column_order = []
        if save_index: column_order.append('sample_index')
        column_order.extend(['true_label_sequence', 'predicted_sequence'])
        df = pd.DataFrame(csv_data)[column_order]

    else:
        # --- 模式 B: 完全展开保存 ---
        print("Saving in 'expanded' mode.")
        num_samples, seq_len = all_preds_array.shape

        # 使用 np.repeat 和 np.tile 来高效地生成索引
        # 样本索引: [0, 0, ..., 0, 1, 1, ..., 1, ..., N-1, ...]
        sample_indices = np.repeat(np.arange(num_samples), seq_len)
        # 时间步索引: [0, 1, ..., L-1, 0, 1, ..., L-1, ...]
        timestep_indices = np.tile(np.arange(seq_len), num_samples)

        # 将预测和标签数组展平
        flat_preds = all_preds_array.flatten()
        flat_labels = all_labels_array.flatten()

        # 创建DataFrame
        df = pd.DataFrame({
            'sample_index': sample_indices,
            'timestep_index': timestep_indices,
            'true_label': flat_labels,
            'predicted_value': flat_preds
        })
    try:
        df.to_csv(save_path, index=False, float_format='%.6f')  # float_format 保证数字精度
        print(f"Predictions successfully saved to: {save_path}")
    except Exception as e:
        print(f"Error saving file to {save_path}: {e}")


def do_predict(train_log_dir, data_base, outputs_dir, device="cuda"):
    train_log_dir = os.path.abspath(train_log_dir)
    if os.path.exists(train_log_dir) and os.path.isdir(train_log_dir):
        run_config_file = os.path.join(train_log_dir, "run_config.json")
        weight_path = os.path.join(train_log_dir, "weights", 'model.pth')
        test_log_path = os.path.join(train_log_dir, "test_result.csv")
        if all([os.path.exists(run_config_file), os.path.exists(weight_path), os.path.exists(test_log_path)]):
            with open(run_config_file, "r", encoding="utf8") as conf_file:
                run_config = json.load(conf_file)
                model_name = run_config["model_name"]
                data_name = run_config["data_name"]
                max_len = run_config["max_len"]
                batch_size = run_config["batch_size"]
                dataset_dir = os.path.join(data_base, data_name)
                data_set = DirDataset(dataset_dir)
                assert data_name == data_set.name
                data_set.recheck_data(folds=run_config["folds"], max_len=max_len)
                model_config = get_model_default_params(model_name)
                model_config.update(run_config)
                model = get_model(model_name, data_name, model_config)
                feature_names, label_names = model.inputs_specs
                trdata_loader, vadata_loader, tedata_loader = data_set.split_generator(dataset_dir, feature_names,
                                                                                       label_names,
                                                                                       max_len=max_len)
                test_dataloader = DataLoader(tedata_loader, batch_size=batch_size)
                stat_dict = torch.load(weight_path)
                model.load_state_dict(stat_dict)
                model.to(device)
                all_labels = []
                all_preds = []
                print(f"predict for model {model_name} on dataset {data_name}")
                predict_outputs_dir=os.path.join(outputs_dir,f'{model_name}_{data_name}')
                if not os.path.exists(predict_outputs_dir):
                    os.makedirs(predict_outputs_dir)
                with torch.no_grad():
                    for step, data in enumerate(test_dataloader):
                        print(f"predict for model {model_name} on dataset {data_name} step {step}")
                        predict_batch = os.path.join(predict_outputs_dir, f"batch_{step}.csv")
                        data = to_device(data, device)
                        model.test_step(data)
                        x, y, mask, sample_weight = model.data_map(data)
                        # Compute prediction error
                        outputs = model.forward(x, training=False, mask=mask)
                        if isinstance(outputs, (list, tuple)):
                            y_pred = outputs[0]
                        else:
                            y_pred = outputs
                        if y.shape[1] < mask.shape[1]:
                            mask = mask[:, 1:]
                        # y_pred, y = y_pred.masked_select(mask), y.masked_select(mask)
                        # y_pred, y = y_pred.detach().cpu().numpy(), y.detach().cpu().numpy()
                        # mask = mask[:, 1:, :]
                        mask = mask.reshape(-1).detach().cpu().numpy()
                        pad = (mask - 1) * np.ones_like(mask)
                        y_pred = y_pred.reshape(-1).detach().cpu().numpy()
                        y_pred = y_pred * mask + pad
                        # q = q[:, 1:].reshape(-1).detach().cpu().numpy()
                        # q = q * mask + pad
                        y = y.reshape(-1).detach().cpu().numpy()

                        predcit_datas = { "response": y.tolist(), "predict": y_pred.tolist()}
                        pd.DataFrame(predcit_datas).to_csv(predict_batch, index=False, encoding='utf-8',
                                                           float_format='%.6f')
                #         all_labels.append(y.squeeze())
                #         all_preds.append(y_pred.squeeze())
                # output_file = os.path.join(outputs_base_dir, f"{data_name}-{model_name}.csv")
                # save_predictions_to_csv(output_file, all_preds, all_labels, expand_sequences=True)

        else:
            print(f"files in {train_log_dir} has error")


if __name__ == "__main__":
    args = parser.parse_args()
    outputs_base_dir = os.path.abspath(args.result_base)
    log_dir = os.path.abspath(args.log_dir)
    if not os.path.exists(outputs_base_dir):
        os.makedirs(outputs_base_dir, exist_ok=True)
    # for d in os.listdir(log_dir):
    #     do_predict(os.path.join(log_dir, d), args.data_base, outputs_base_dir, args.device)
    do_predict(log_dir, args.data_base, outputs_base_dir, args.device)
