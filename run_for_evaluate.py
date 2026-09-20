#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 2026/2/3 0:19
# @Author: hb925
# @File: run_for_evaluate.py
import argparse
import json
import os
import os.path
import random

import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from torch.utils.data import DataLoader

from torch_kt.data_loader import DirDataset
from torch_kt.runner import get_model, get_model_default_params, to_device


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser(description='evaluate for model')
parser.add_argument('log_dir', nargs='?',default='logs', help='Model train log dir')
parser.add_argument('--data_base', default='./data', help='logs dir path')
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda', 'npu'])

def compute_metrics(all_labels,all_preds):
    prelabels = [1 if p >= 0.5 else 0 for p in all_preds]
    auc = metrics.roc_auc_score(all_labels, all_preds)
    acc = metrics.accuracy_score(all_labels, prelabels)
    rmse = metrics.mean_squared_error(all_labels, all_preds, squared=False)
    return {"auc": auc, "acc": acc, "rmse": rmse}
def do_evaluate(train_log_dir, data_base,  device="cuda"):
    train_log_dir = os.path.abspath(train_log_dir)
    if os.path.exists(train_log_dir) and os.path.isdir(train_log_dir):
        run_config_file = os.path.join(train_log_dir, "run_config.json")
        test_log_path = os.path.join(train_log_dir, "test_result.csv")
        if all([os.path.exists(run_config_file),  os.path.exists(test_log_path)]):
            with open(run_config_file, "r", encoding="utf8") as conf_file:
                run_config = json.load(conf_file)
                model_name = run_config["model_name"]
                data_name = run_config["data_name"]
                max_len = run_config["max_len"]
                batch_size = run_config["batch_size"]
                folds=run_config["folds"]
                dataset_dir = os.path.join(data_base, data_name)
                data_set = DirDataset(dataset_dir)
                assert data_name == data_set.name
                data_set.recheck_data(folds=folds, max_len=max_len)
                if folds > 0:
                    folds_params = [(os.path.join(dataset_dir, f'k{k}'), os.path.join(train_log_dir, f'k{k}')) for k in
                                    range(folds)]
                else:
                    folds_params = [(dataset_dir, train_log_dir)]
                model_config = get_model_default_params(model_name)
                model_config.update(run_config)
                all_results = []
                for datas_dir, logs_dir in folds_params:
                    model = get_model(model_name, data_name, model_config)
                    feature_names, label_names = model.inputs_specs
                    trdata_loader, vadata_loader, tedata_loader = data_set.split_generator(datas_dir, feature_names,
                                                                                           label_names,
                                                                                           max_len=max_len)
                    weight_path = os.path.join(logs_dir, "weights", 'model.pth')
                    test_dataloader = DataLoader(tedata_loader, batch_size=batch_size)
                    stat_dict = torch.load(weight_path)
                    model.load_state_dict(stat_dict,strict=False)
                    model.to(device)
                    all_labels = []
                    all_preds = []
                    all_labels_diff = []
                    all_preds_diff = []
                    all_labels_diff_n = []
                    all_preds_diff_n = []
                    print(f"evaluate for model {model_name} on dataset {data_name}")
                    with torch.no_grad():
                        for step, data in enumerate(test_dataloader):
                            print(f"evaluate for model {model_name} on dataset {data_name} step {step}")
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
                            r_eq_mask = torch.eq(y[:, 1:, ], y[:, :-1, ])
                            r_eq_mask_e = torch.concat([torch.ones_like(r_eq_mask[:, :1]), r_eq_mask], dim=1)
                            r_eq_mask_e=(~r_eq_mask_e)
                            mask_d=mask*r_eq_mask_e
                            mask_d_n=mask*torch.concat([r_eq_mask_e[:, :1], r_eq_mask_e[:,:-1]], dim=1)
                            y_pred_a, y_a = y_pred.masked_select(mask), y.masked_select(mask)
                            y_pred_a, y_a = y_pred_a.detach(), y_a.detach()
                            all_labels.append(y_a)
                            all_preds.append(y_pred_a)

                            y_pred_diff, y_diff = y_pred.masked_select(mask_d), y.masked_select(mask_d)
                            y_pred_diff, y_diff = y_pred_diff.detach(), y_diff.detach()
                            all_labels_diff.append(y_diff)
                            all_preds_diff.append(y_pred_diff)

                            y_pred_diff_n, y_diff_n = y_pred.masked_select(mask_d_n), y.masked_select(mask_d_n)
                            y_pred_diff_n, y_diff_n = y_pred_diff_n.detach(), y_diff_n.detach()
                            all_labels_diff_n.append(y_diff_n)
                            all_preds_diff_n.append(y_pred_diff_n)
                        ts = torch.cat(all_labels, dim=0).cpu().numpy()
                        ps = torch.cat(all_preds, dim=0).cpu().numpy()
                        ts_diff = torch.cat(all_labels_diff, dim=0).cpu().numpy()
                        ps_diff = torch.cat(all_preds_diff, dim=0).cpu().numpy()
                        ts_diff_n = torch.cat(all_labels_diff_n, dim=0).cpu().numpy()
                        ps_diff_n = torch.cat(all_preds_diff_n, dim=0).cpu().numpy()
                        evalue_metric=compute_metrics(ts,ps)
                        evalue_metric_diff = compute_metrics(ts_diff, ps_diff)
                        evalue_metric_diff_n = compute_metrics(ts_diff_n, ps_diff_n)
                        for k,v in evalue_metric_diff.items():
                            evalue_metric[f'diff_{k}']=v
                        for k,v in evalue_metric_diff_n.items():
                            evalue_metric[f'diff_n_{k}']=v
                        all_results.append(evalue_metric)
                df = pd.DataFrame(all_results)
                mean_result = df.mean().to_dict()
                for metric in df.columns:
                    mean_result[f"p_{metric}"] = (
                        df[metric].std() if len(df) > 1 and metric in df.columns else -1
                    )
                mean_result.update({"model_name":model_name,"data_name":data_name})
                return mean_result
        else:
            print(f"files in {train_log_dir} has error")
            return None


if __name__ == "__main__":
    args = parser.parse_args()
    log_dir = os.path.abspath(args.log_dir)
    # evaluate_result = do_evaluate(log_dir, args.data_base, args.device)
    all_evaluate_results=[]
    for d in os.listdir(log_dir):
        evaluate_result=do_evaluate(os.path.join(log_dir, d), args.data_base, args.device)
        if evaluate_result is not None:
            all_evaluate_results.append(evaluate_result)
    dir_parts=os.path.split(log_dir)
    pd.DataFrame(all_evaluate_results).to_csv(f"{dir_parts[-1]}_evaluate_results.csv",index=False, encoding='utf-8', float_format='%.5f')