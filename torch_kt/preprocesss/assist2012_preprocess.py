#!/usr/bin/env python
# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 0011 下午 9:36
# @Author  : hb
# @File    : assist2012_preprocess.py
import os

import numpy as np
import pandas as pd

from .utils import to_seq_data, time2ts, parent_dir, ORDER_KEY, \
    ALL_RAW_FILE_NAME, rename_dataset

DATASET_NAME = "assist2012"
USE_COLS = ["user_id", "skill_id", "start_time", "problem_id", "correct", "ms_first_response"]
SORT_COLS = ['start_time']


def process_raw_data(org_file, out_dir,  **kwargs):
    if not os.path.exists(org_file):
        base_dir = parent_dir(org_file)
        data_file = os.path.join(base_dir, "2012-2013-data-with-predictions-4-final.csv")
        df = pd.read_csv(data_file, encoding="utf-8", encoding_errors="ignore", dtype={"skill_id": str},
                         low_memory=False)
        print(df.dtypes.to_dict())
        df = df.astype({"correct": np.int64}, copy=False).astype({"correct": np.int8}, copy=False)
        df.to_csv(org_file, encoding="utf-8", index=False)
    dataset_dir = os.path.join(out_dir, rename_dataset(DATASET_NAME,**kwargs))
    os.makedirs(dataset_dir, exist_ok=True)
    df = pd.read_csv(org_file, encoding="utf-8", encoding_errors="ignore", dtype={"correct": np.int8, "skill_id": str},
                     low_memory=False)
    miss_cols = set(USE_COLS) - set(df.columns)
    if len(miss_cols) > 0:
        raise KeyError(f"The column {','.join(miss_cols)} was not found on {org_file}")
    df.sort_values(by=SORT_COLS, inplace=True)
    df[ORDER_KEY] = df.index.values
    df['start_timestamp'] = df['start_time'].apply(time2ts)
    df.drop(columns=['start_time'], inplace=True)
    df.rename(columns={"ms_first_response": 'response_ms'}, inplace=True)
    fn = os.path.join(dataset_dir, ALL_RAW_FILE_NAME)
    df.to_csv(fn, encoding="utf-8", index=False)
    seq, dataset_info, dataset_dir = to_seq_data(fn, dataset_dir, DATASET_NAME, **kwargs)
    return seq, dataset_info, dataset_dir
