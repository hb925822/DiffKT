#!/usr/bin/env python
# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/6/28 0028 上午 12:15
# @Author  : hb
# @File    : algebra2005_preprocess.py
import os

import numpy as np
import pandas as pd

from .utils import parent_dir, to_seq_data, ORDER_KEY, ALL_RAW_FILE_NAME, time2ts, rename_dataset

DATASET_NAME = "algebra2005"
USE_COLS = ["Anon Student Id", "Problem Name", "Step Name", "Step Start Time", "Step End Time", "KC(Default)",
            "Correct First Attempt", "First Transaction Time"]
COL_MAPS = {"Anon Student Id": "user_id", "KC(Default)": "skill_id", "Questions": "problem_id",
            "Correct First Attempt": "correct"}
SORT_COLS = ["First Transaction Time"]


def process_raw_data(fn, out_dir,  **kwargs):
    dataset_dir = os.path.join(out_dir, rename_dataset(DATASET_NAME,**kwargs))
    os.makedirs(dataset_dir, exist_ok=True)
    if not os.path.exists(fn):
        base_dir = parent_dir(fn)
        train_file = os.path.join(base_dir, "algebra_2005_2006_train.txt")
        test_file = os.path.join(base_dir, "algebra_2005_2006_test.txt")
        tr_df = pd.read_csv(train_file, encoding="utf-8", delimiter="\t", encoding_errors="ignore", usecols=USE_COLS,
                            low_memory=False)
        te_df = pd.read_csv(test_file, encoding="utf-8", delimiter="\t", encoding_errors="ignore", usecols=USE_COLS,
                            low_memory=False)
        all_df = pd.concat([tr_df, te_df])
        all_df = all_df.dropna(
            subset=["Anon Student Id", "KC(Default)", "Correct First Attempt", "First Transaction Time"])
        replace_text = lambda x: x.replace("_", "#").replace(",", "@")#lambda x: x.replace("_", "####").replace(",", "@@@@")#
        all_df["KC(Default)"] = all_df["KC(Default)"].apply(replace_text)
        all_df["KC(Default)"] = all_df["KC(Default)"].apply(lambda x: x.replace("~~", "_"))
        #以‘_’分割将知识点拆分后总知识点个数是112
        all_df["First Transaction Time"] = all_df["First Transaction Time"].apply(time2ts)
        all_df["Problem Name"] = all_df["Problem Name"].apply(replace_text)
        all_df["Step Name"] = all_df["Step Name"].apply(replace_text)
        all_df["Questions"] = all_df.apply(lambda x: f"{x['Problem Name']}--{x['Step Name']}", axis=1)
        if not all_df["Correct First Attempt"].hasnans:
            all_df = all_df.astype({"Correct First Attempt": np.int8})
        print(all_df.dtypes.to_dict())
        all_df.to_csv(fn, encoding="utf-8", index=False)
    df = pd.read_csv(fn, encoding="utf-8", encoding_errors="ignore", dtype={"Correct First Attempt": np.int8},low_memory=False)
    miss_cols = set(USE_COLS) - set(df.columns)
    if len(miss_cols) > 0:
        raise KeyError(f"The column {','.join(miss_cols)} was not found on {fn}")
    df.sort_values(by=SORT_COLS, inplace=True, ignore_index=True)
    df[ORDER_KEY] = df.index.values
    df.rename(columns=COL_MAPS, inplace=True)
    df.rename(columns={"First Transaction Time": 'start_timestamp'}, inplace=True)
    fn = os.path.join(dataset_dir, ALL_RAW_FILE_NAME)
    df.to_csv(fn, encoding="utf-8", index=False)
    seq, dataset_info, dataset_dir = to_seq_data(fn, dataset_dir, DATASET_NAME, **kwargs)
    return seq, dataset_info, dataset_dir
