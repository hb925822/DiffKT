# _*_ coding: utf-8 _*_
# @Time : 2023/3/29 11:54
# @Author : hb
# @File : assist0910_preprocess.py
# @desc :
import os

import pandas as pd

from .utils import parent_dir, to_seq_data, ORDER_KEY, ALL_RAW_FILE_NAME, rename_dataset

DATASET_NAME = "assist0910"

USE_COLS = ["user_id", "skill_id", "problem_id", "correct", "order_id"]
COLS_TYPE = {"skill_id": str}
SORT_COLS = ["order_id"]


def process_raw_data(org_file, out_dir, **kwargs):
    if not os.path.exists(org_file):
        base_dir = parent_dir(org_file)
        data_file = os.path.join(base_dir, "skill_builder_data_corrected.csv")
        df = pd.read_csv(data_file, encoding="utf-8", encoding_errors="ignore", dtype=COLS_TYPE, low_memory=False)
        df.drop_duplicates(subset=["order_id"], inplace=True)
        df = df[df['original'].isin([1])]
        print(df.dtypes.to_dict())
        df.to_csv(org_file, encoding="utf-8", index=False)
    dataset_dir = os.path.join(out_dir, rename_dataset(DATASET_NAME, **kwargs))
    os.makedirs(dataset_dir, exist_ok=True)
    df = pd.read_csv(org_file, encoding="utf-8", encoding_errors="ignore", dtype=COLS_TYPE, low_memory=False)
    miss_cols = set(USE_COLS) - set(df.columns)
    if len(miss_cols) > 0:
        raise KeyError(f"The column {','.join(miss_cols)} was not found on {org_file}")
    df.sort_values(by=SORT_COLS, inplace=True, ignore_index=True)
    df[ORDER_KEY] = df.index.values
    fn = os.path.join(dataset_dir, ALL_RAW_FILE_NAME)
    df.to_csv(fn, encoding="utf-8", index=False)
    seq, dataset_info, dataset_dir = to_seq_data(fn, dataset_dir, DATASET_NAME, **kwargs)
    return seq, dataset_info, dataset_dir
