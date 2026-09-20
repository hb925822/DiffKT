#!/usr/bin/env python
# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/4/27 0027 下午 11:43
# @Author  : hb
# @File    : assist2017_preprocess.py
import os

import pandas as pd

from .utils import to_seq_data, parent_dir, ORDER_KEY, ALL_RAW_FILE_NAME, rename_dataset

DATASET_NAME = "assist2017"
USE_COLS = ["studentId", "problemId", "skill", "correct", "timeTaken", "startTime"]
SORT_COLS = ['startTime']
COL_MAPS = {"studentId": "user_id", "skill": "skill_id", "problemId": "problem_id", }


def process_raw_data(org_file, out_dir,  **kwargs):
    if not os.path.exists(org_file):
        base_dir = parent_dir(org_file)
        data_file = os.path.join(base_dir, "anonymized_full_release_competition_dataset.csv")
        df = pd.read_csv(data_file, encoding="utf-8", encoding_errors="ignore", dtype={"skill": str}, low_memory=False)
        print(df.dtypes.to_dict())
        df.to_csv(org_file, encoding="utf-8", index=False)
    dataset_dir = os.path.join(out_dir, rename_dataset(DATASET_NAME,**kwargs))
    os.makedirs(dataset_dir, exist_ok=True)
    df = pd.read_csv(org_file, encoding="utf-8", encoding_errors="ignore", low_memory=False)
    miss_cols = set(USE_COLS) - set(df.columns)
    if len(miss_cols) > 0:
        raise KeyError(f"The column {','.join(miss_cols)} was not found on {org_file}")
    df['response_ms'] = df['timeTaken'].apply(lambda x: round(x * 1000))
    df['start_timestamp'] = df['startTime'].apply(lambda x: round(x * 1000))
    df.sort_values(by=SORT_COLS, inplace=True)
    df.sort_index(inplace=True, ignore_index=True)
    df[ORDER_KEY] = df.index.values
    df.rename(columns=COL_MAPS, inplace=True)
    fn = os.path.join(dataset_dir, ALL_RAW_FILE_NAME)
    df.to_csv(fn, encoding="utf-8", index=False)
    seq, dataset_info, dataset_dir = to_seq_data(fn, dataset_dir, DATASET_NAME, **kwargs)
    return seq, dataset_info, dataset_dir
