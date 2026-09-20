#!/usr/bin/env python
# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/6/27 0027 下午 3:49
# @Author  : hb
# @File    : junyi_preprocess.py
import os

import numpy as np
import pandas as pd

from .utils import parent_dir, to_seq_data, ALL_RAW_FILE_NAME, ORDER_KEY, rename_dataset

DATASET_NAME = "junyi"
USE_COLS = ['user_id', 'exercise', 'time_done', 'topic', 'correct', "time_taken_attempts"]
COL_MAPS = {"topic": "skill_id", "exercise": "problem_id"}
SORT_COLS = ["time_done"]


def _load_qc_map(filename):
    df = pd.read_csv(filename, encoding="utf-8", low_memory=False).dropna(subset=["name", "topic"])
    dq2c = dict()
    for name, topic in zip(df["name"], df["topic"]):
        if name not in dq2c:
            dq2c[name] = topic
        else:
            print(f"already has topic in dict: {name}: {topic}, {dq2c[name]}")
    print(f"dq2c: {len(dq2c)}")
    return dq2c


def str_value_map(x):
    if x is None:
        return x
    else:
        return x.replace("_", "####").replace(",", "@@@@")


def trans_time_taken(x):
    if pd.isna(x):
        return -1
    else:
        return int(x.split("&")[0]) * 1000


def process_raw_data(fn, out_dir,  **kwargs):
    if not os.path.exists(fn):
        base_dir = parent_dir(fn)
        data_file = os.path.join(base_dir, "junyi_ProblemLog_original.csv")
        map_file = os.path.join(base_dir, "junyi_Exercise_table.csv")
        qc_map = _load_qc_map(map_file)
        df = pd.read_csv(data_file, encoding="utf-8", encoding_errors="ignore", low_memory=False)
        df["topic"] = df["exercise"].apply(lambda q: None if q not in qc_map else qc_map[q])
        df["exercise"] = df["exercise"].apply(str_value_map)
        df["topic"] = df["topic"].apply(str_value_map)
        df["correct"] = df["correct"].apply(np.int8)
        print(df.dtypes.to_dict())
        df.to_csv(fn, encoding="utf-8", index=False)
    dataset_dir = os.path.join(out_dir, rename_dataset(DATASET_NAME,**kwargs))
    os.makedirs(dataset_dir, exist_ok=True)
    df = pd.read_csv(fn, encoding="utf-8", encoding_errors="ignore", low_memory=False)
    df.sort_values(by=SORT_COLS, inplace=True, ignore_index=True)
    df[ORDER_KEY] = df.index.values
    df.rename(columns=COL_MAPS, inplace=True)
    df["start_timestamp"] = df["time_done"].apply(lambda x: round(int(x) / 1000))
    df["response_ms"] = df["time_taken_attempts"].apply(trans_time_taken)
    df.drop(columns=["time_done", "time_taken_attempts"])
    fn = os.path.join(dataset_dir, ALL_RAW_FILE_NAME)
    df.to_csv(fn, encoding="utf-8", index=False)
    seq, dataset_info, dataset_dir = to_seq_data(fn, dataset_dir, DATASET_NAME, **kwargs)
    return seq, dataset_info, dataset_dir
