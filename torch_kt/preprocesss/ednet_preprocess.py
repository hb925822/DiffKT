#!/usr/bin/env python
# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/6/27 0027 下午 4:22
# @Author  : hb
# @File    : ednet_preprocess.py
import os
import pandas as pd

from .utils import parent_dir, ALL_RAW_FILE_NAME, to_seq_data, ORDER_KEY, rename_dataset

DATASET_NAME = "ednet"
USE_COLS = ["user_id", "timestamp", "question_id", "tags", "correct", "elapsed_time"]
COL_MAPS = {"tags": "skill_id", "question_id": "problem_id", "timestamp": "start_timestamp",
            "elapsed_time": 'response_ms'}
SORT_COLS = ["timestamp"]


def _merge_rawdata(file_dir, question_file, tmp_file):
    ca = pd.read_csv(question_file)
    ca['tags'] = ca['tags'].apply(lambda x: x.replace(";", "_"))
    ca = ca[ca['tags'] != '-1']
    file_lists = []
    count = 0
    if os.path.exists(tmp_file):
        os.remove(tmp_file)
    for f in os.listdir(file_dir):
        if ".csv" in f:
            user_file = os.path.join(file_dir, f)
            df = pd.read_csv(user_file, encoding="utf-8", encoding_errors="ignore")
            df['user_id'] = f[1:-4]
            df = df.merge(ca, sort=False, how='left')
            df["correct"] = df.apply(
                lambda x: 1 if str(x["correct_answer"]).strip() == str(x["user_answer"]).strip() else 0, axis=1)
            df["elapsed_time"] = df["elapsed_time"].apply(lambda x: round(x))
            # file_lists.append(df[["user_id","timestamp","question_id","tags","correct","elapsed_time"]])
            df = df[["user_id", "timestamp", "question_id", "tags", "correct", "elapsed_time"]]
            count += 1
            if df.isna().values.any() or df.isnull().values.any():
                continue
            if len(df) > 300:
                df = df.iloc[:300]
            else:
                continue
            if os.path.exists(tmp_file):
                df.to_csv(tmp_file, index=False, header=False, encoding="utf-8", mode="a")
            else:
                df.to_csv(tmp_file, index=False, encoding="utf-8")
    print(f'total user nums:{count}')
    # if count>100:
    #     break
    # all_sa = pd.concat(file_lists)
    # all_sa.to_csv(tmp_file, index=False,encoding="utf-8")


def process_raw_data(fn, out_dir, **kwargs):
    if not os.path.exists(fn):
        base_dir = parent_dir(fn)
        file_dir = os.path.join(base_dir, "KT1")
        question_file = os.path.join(base_dir, "contents", "questions.csv")
        _merge_rawdata(file_dir, question_file, fn)
    dataset_dir = os.path.join(out_dir, rename_dataset(DATASET_NAME, **kwargs))
    os.makedirs(dataset_dir, exist_ok=True)
    df = pd.read_csv(fn, encoding="utf-8", encoding_errors="ignore", low_memory=False)
    df.sort_values(by=SORT_COLS, inplace=True, ignore_index=True)
    df[ORDER_KEY] = df.index.values
    df.rename(columns=COL_MAPS, inplace=True)
    fn = os.path.join(dataset_dir, ALL_RAW_FILE_NAME)
    df.to_csv(fn, encoding="utf-8", index=False)
    seq, dataset_info, dataset_dir = to_seq_data(fn, dataset_dir, DATASET_NAME, **kwargs)
    return seq, dataset_info, dataset_dir
