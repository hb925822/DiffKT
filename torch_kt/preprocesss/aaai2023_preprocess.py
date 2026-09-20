#!/usr/bin/env python
# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/6/28 0028 下午 1:52
# @Author  : hb
# @File    : aaai2023_preprocess.py
import os
import pandas as pd
import numpy as np
from .utils import parent_dir, to_seq_data, ORDER_KEY, ALL_RAW_FILE_NAME, rename_dataset

DATASET_NAME = "aaai2023"
USE_COLS = ["uid", "concepts", "questions", "timestamps", "responses"]
SORT_COLS = ["timestamps"]
COL_MAPS = {"uid": "user_id", "concepts": "skill_id", "questions": "problem_id", "responses": "correct",
            "timestamps": "start_timestamp"}


def split_func(x):
    questions = x['questions'].split(',')
    concepts = x['concepts'].split(',')
    responses = x['responses'].split(',')
    selectmasks = x['selectmasks'].split(',')
    timestamps = x['timestamps'].split(',')
    is_repeat = x['is_repeat'].split(',')
    tmp = pd.DataFrame({'questions': questions, 'concepts': concepts, 'responses': responses,
                        'timestamps': timestamps, 'selectmasks': selectmasks, 'is_repeat': is_repeat})
    tmp["uid"] = x["uid"]
    tmp = tmp[tmp["selectmasks"] == '1']
    tmp.drop(columns=["selectmasks"], inplace=True)
    return tmp


def split_func1(x):
    questions = x['questions'].split(',')
    concepts = x['concepts'].split(',')
    responses = x['responses'].split(',')
    timestamps = x['timestamps'].split(',')
    is_repeat = x['is_repeat'].split(',')
    tmp = pd.DataFrame({'questions': questions, 'concepts': concepts, 'responses': responses,
                        'timestamps': timestamps, 'is_repeat': is_repeat})
    tmp["uid"] = x["uid"]
    tmp = tmp[tmp["responses"] != '-1']
    return tmp


def process_raw_data(org_file, out_dir, **kwargs):
    dataset_dir = os.path.join(out_dir, rename_dataset(DATASET_NAME, **kwargs))
    os.makedirs(dataset_dir, exist_ok=True)
    if not os.path.exists(org_file):
        base_dir = parent_dir(org_file)
        data_file = os.path.join(base_dir, "train_valid_sequences.csv")
        test_data_file = os.path.join(base_dir, "pykt_test.csv")
        seq_df = pd.read_csv(data_file, encoding="utf-8", encoding_errors="ignore", low_memory=False)
        seq_df.drop(columns=["fold"], inplace=True)
        test_seq_df = pd.read_csv(test_data_file, encoding="utf-8", encoding_errors="ignore", low_memory=False)
        sub_fras = seq_df.apply(split_func, axis=1).to_list()
        sub_fras1 = test_seq_df.apply(split_func1, axis=1).to_list()
        df = pd.concat(sub_fras + sub_fras1).reset_index(drop=True)
        df = df.astype({"responses": np.int8, "timestamps": np.int64})
        print(df.dtypes.to_dict())
        df.to_csv(org_file, encoding="utf-8", index=False)
    df = pd.read_csv(org_file, encoding="utf-8", encoding_errors="ignore", low_memory=False)

    miss_cols = set(USE_COLS) - set(df.columns)
    if len(miss_cols) > 0:
        raise KeyError(f"The column {','.join(miss_cols)} was not found on {org_file}")
    df.sort_values(by=SORT_COLS, inplace=True, ignore_index=True)
    df["start_timestamp"] = df["timestamps"].astype(np.int64, copy=False)
    df[ORDER_KEY] = df.index.values
    df.rename(columns=COL_MAPS, inplace=True)
    fn = os.path.join(dataset_dir, ALL_RAW_FILE_NAME)
    df.to_csv(fn, encoding="utf-8", index=False)
    seq, dataset_info, dataset_dir = to_seq_data(fn, dataset_dir, DATASET_NAME, **kwargs)
    return seq, dataset_info, dataset_dir
