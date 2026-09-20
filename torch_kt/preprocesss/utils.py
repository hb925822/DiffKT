# _*_ coding: utf-8 _*_
# @Time : 2023/4/6 21:12
# @Author : hb
# @File : wandb_utils.py
# @desc :
import collections
import json
import os
import random
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd

ORDER_KEY = 'order_index'
ALL_RAW_FILE_NAME = 'all_raw.csv'
NULL_Fill_VALUE = "null"


def parent_dir(file_dir):
    dir_parts = os.path.split(file_dir)
    _parent_dir = os.path.join("", *dir_parts[:-1])
    return _parent_dir


def do_statistics(df, file_path=None, has_problem=True):
    col_state_value = {}
    u_counts_data = df["user_id"].value_counts(dropna=False)
    user_interaction = u_counts_data.agg(["mean", "max", "min", "sum", "count"]).to_frame(name="user")

    s_counts_data = df["skill_id"].value_counts(dropna=False)
    skill_interaction = s_counts_data.agg(["mean", "max", "min", "sum", "count"]).to_frame(name="skill")

    us_counts_data = df[["user_id", "skill_id"]].value_counts(dropna=False)
    user_skill_interaction = us_counts_data.agg(["mean", "max", "min", "sum", "count"]).to_frame(name="user_skill")

    interaction_values = [user_interaction, skill_interaction, user_skill_interaction, ]
    col_state_value["user_interaction_v"] = u_counts_data.to_frame(name="count").reset_index(names="user").to_dict(
        "list")
    col_state_value["skill_interaction_v"] = s_counts_data.to_frame(name="count").reset_index(names="skill").to_dict(
        "list")
    if has_problem:
        p_counts_data = df["problem_id"].value_counts(dropna=False)
        problem_interaction = p_counts_data.agg(["mean", "max", "min", "sum", "count"]).to_frame(name="problem")
        up_counts_data = df[["user_id", "problem_id"]].value_counts(dropna=False)
        user_problem_interaction = up_counts_data.agg(["mean", "max", "min", "sum", "count"]).to_frame(
            name="user_problem")

        sp_counts_data = df[["skill_id", "problem_id"]].value_counts(dropna=False)
        skill_problem_interaction = sp_counts_data.agg(["mean", "max", "min", "sum", "count"]).to_frame(
            name="problem_skill")
        interaction_values.extend([problem_interaction, user_problem_interaction, skill_problem_interaction])
        col_state_value["problem_interaction_v"] = p_counts_data.to_frame(name="count").reset_index(
            names="problem").to_dict("list")
    all_interaction = pd.concat(interaction_values, axis=1).reset_index().rename(columns={"index": "value"})
    col_state_value["interaction_statistics"] = all_interaction.to_dict("list")
    # col_state_value["skill_problem_count"] = df.groupby(["skill_id"]).apply(
    #     lambda x: pd.Series({"count": len(x["problem_id"].unique())})).reset_index(
    #     names="skill_id").to_dict("list")
    # if has_problem:
    # col_state_value["problem_skill_count"] = df.groupby(["problem_id"]).apply(
    #     lambda x: pd.Series({"count": len(x["skill_id"].unique())})).reset_index(
    #     names="problem_id").to_dict("list")

    if file_path:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(col_state_value, f, indent=4)
    return col_state_value


def do_response_statistics(df, dataset_dir, has_problem=True):
    col_state_value = {}
    skill_response = df.groupby("skill")["correct"].agg(count="count", correct="sum").reset_index(names="skill")
    col_state_value["skill_response"] = skill_response.to_dict("list")
    if has_problem:
        problem_response = df.groupby("problem")["correct"].agg(count="count", correct="sum").reset_index(
            names="problem")
        skill_problem_response = df.groupby(["skill"]).apply(
            lambda x: x.groupby("problem")["correct"].agg(count="count", correct="sum")).reset_index()

        def count_do(x, name):
            values = x[name].unique()
            vs = [str(x) for x in values]
            return pd.Series(
                {"count": len(values), name + "s": vs})

        skill_problem_count = df.groupby(["skill"]).apply(
            count_do, "problem").reset_index(names="skill")
        problem_skill_count = df.groupby(["problem"]).apply(
            count_do, "skill").reset_index(names="problem")
        col_state_value["skill_problem_response"] = skill_problem_response.to_dict("list")
        col_state_value["skill_problem_count"] = skill_problem_count.to_dict("list")
        col_state_value["problem_skill_count"] = problem_skill_count.to_dict("list")
        col_state_value["problem_response"] = problem_response.to_dict("list")
    if dataset_dir:
        f_skill_response = os.path.join(dataset_dir, "skill_response.json")
        with open(f_skill_response, "w", encoding="utf-8") as f:
            json.dump(col_state_value["skill_response"], f, indent=4)
        if has_problem:
            f_problem_response = os.path.join(dataset_dir, "problem_response.json")
            f_skill_problem_response = os.path.join(dataset_dir, "skill_problem_response.json")
            f_skill_problem_count = os.path.join(dataset_dir, "skill_problem_count.json")
            f_problem_skill_count = os.path.join(dataset_dir, "problem_skill_count.json")
            with open(f_problem_response, "w", encoding="utf-8") as f:
                json.dump(col_state_value["problem_response"], f, indent=4)
            with open(f_skill_problem_response, "w", encoding="utf-8") as f:
                json.dump(col_state_value["skill_problem_response"], f, indent=4)
            with open(f_skill_problem_count, "w", encoding="utf-8") as f:
                json.dump(col_state_value["skill_problem_count"], f, indent=4)
            with open(f_problem_skill_count, "w", encoding="utf-8") as f:
                json.dump(col_state_value["problem_skill_count"], f, indent=4)
    return col_state_value


# def compose_dataset():
#

def _gen_fold_index(data_lens, folds):
    fold_size = data_lens // folds
    remains = data_lens - fold_size * folds
    folds_set = []
    for i in range(folds):
        folds_set.extend([i for _ in range(fold_size)])
        print(f'fold:{i},len:{fold_size}')
    if remains > 0:
        folds_set.extend([int(k % 5) for k in range(remains)])
    random.shuffle(folds_set)
    random.shuffle(folds_set)
    random.shuffle(folds_set)
    return folds_set


def save_dataset(train_data, valid_data, test_data, data_info, out_dir, all_datas=None):
    os.makedirs(out_dir, exist_ok=True)
    if train_data is not None:
        train_data.to_pickle(os.path.join(out_dir, "train.pkl"))
    if valid_data is not None:
        valid_data.to_pickle(os.path.join(out_dir, "valid.pkl"))
    if test_data is not None:
        test_data.to_pickle(os.path.join(out_dir, "test.pkl"))
    if all_datas is not None:
        all_datas.to_pickle(os.path.join(out_dir, "all.pkl"))
    with open(os.path.join(out_dir, "info.json"), "w", encoding="utf-8") as f:
        json.dump(data_info, f, indent=4)


# def split_dataset(all_data, data_info, dataset_out_dir, test_frac=0.2, val_frac=0.1, folds=5, max_len=-1, min_len=3):
#     dataset_info_split = {'test_frac': test_frac, 'val_frac': val_frac, 'folds': folds}
#     print("############### split train test ################")
#     all_data = all_data.sample(frac=1.0)
#     test_data = all_data.sample(frac=test_frac)
#     val_train_data = all_data[~all_data.index.isin(test_data.index)]
#     data_info.update(dataset_info_split)
#     print("############### split train test over ################")
#     if folds and folds > 0:
#         print("############### split fold ################")
#         data_lens = val_train_data.shape[0]
#         folds_set = _gen_fold_index(data_lens, folds)
#         all_data.loc[val_train_data.index, "fold"] = folds_set
#         all_data.loc[test_data.index, "fold"] = -1
#         all_data = all_data.astype({"fold": np.int32})
#     valid_data = val_train_data.sample(frac=0.1)
#     train_data = val_train_data[~val_train_data.index.isin(valid_data.index)]
#     print("############### split fold over################")
#     if max_len and max_len > 0:
#         print("############### sequence length filter ################")
#         train_data = filter_maxlength(all_data.loc[train_data.index], max_len, min_len)
#         valid_data = filter_maxlength(all_data.loc[valid_data.index], max_len, min_len)
#         test_data = filter_maxlength(all_data.loc[test_data.index], max_len, min_len)
#         data_info.update({"max_len": max_len})
#         print("############### sequence length filter over ################")
#     else:
#         train_data, valid_data, test_data = all_data.loc[train_data.index], all_data.loc[valid_data.index], all_data.loc[
#             test_data.index]
#     print("############### saving  dataname ################")
#     save_dataset(train_data, valid_data, test_data, data_info, dataset_out_dir, all_data)
#     return train_data, valid_data, test_data, data_info
def split_dataset(all_data, data_info, dataset_out_dir, test_frac=0.2, val_frac=0.1, folds=5,keep_split=False):
    dataset_info_split = {'test_frac': test_frac, 'val_frac': val_frac, 'folds': folds}
    old_split=None
    if keep_split:
        old_split_file = os.path.join(dataset_out_dir, "split_fold.pkl")
        if os.path.exists(old_split_file):
            old_split=pd.read_pickle(old_split_file)
        else:
            raise ValueError("no split info file")
        assert all(k in old_split.columns for k in ["user", "split", "fold"])
    if old_split is not None:
        assert len(old_split) == len(all_data)
        all_data = pd.merge(all_data, old_split, how="left", on="user")
        train_data=all_data[all_data["split"]==0]
        valid_data=all_data[all_data["split"]==1]
        test_data=all_data[all_data["split"]==-1]
        new_dataset_info = data_info.copy()
        new_dataset_info.update(dataset_info_split)
    else:
        print("############### split train test ################")
        all_data = all_data.sample(frac=1.0)
        test_data = all_data.sample(frac=test_frac)
        val_train_data = all_data[~all_data.index.isin(test_data.index)]
        data_info.update(dataset_info_split)
        print("############### split train test over ################")
        if folds and folds > 0:
            print("############### split fold ################")
            data_lens = val_train_data.shape[0]
            folds_set = _gen_fold_index(data_lens, folds)
            all_data.loc[val_train_data.index, "fold"] = folds_set
            all_data.loc[test_data.index, "fold"] = -1
            all_data = all_data.astype({"fold": np.int8})
        valid_data = val_train_data.sample(frac=0.1)
        train_data = val_train_data[~val_train_data.index.isin(valid_data.index)]
        all_data.loc[train_data.index, "split"] = 0
        all_data.loc[valid_data.index, "split"] = 1
        all_data.loc[test_data.index, "split"] = -1
        all_data = all_data.astype({"split": np.int8})
        print("############### split fold over################")
        print("############### saving  dataname ################")
    # all_data.to_pickle(os.path.join(dataset_out_dir, "all.pkl"))
    # with open(os.path.join(dataset_out_dir, "info.json"), "w", encoding="utf-8") as f:
    #     json.dump(data_info, f, indent=4)
    save_dataset(train_data, valid_data, test_data, data_info, dataset_out_dir, all_data)
    return data_info


def float2str(x):
    if pd.isna(x):
        return x
    else:
        return str(round(x))


# def float2int(x):
#     if pd.isna(x):
#         return int(-1)
#     else:
#         return round(x)

def rename_dataset(data_name, min_len=10, min_skill_inters=-1, min_problem_inters=-1, max_user_num=-1, **kwargs):
    # if min_len > 0:
    #     data_name += f'ml{min_len}'
    if min_skill_inters > 0:
        data_name += f'-k{min_skill_inters}'
    if min_problem_inters > 0:
        data_name += f'-p{min_problem_inters}'
    if max_user_num > 0:
        data_name += f'-u{max_user_num}'
    return data_name


def to_seq_data(fn, dataset_dir, dataset_name, min_len=10, min_skill_inters=-1, min_problem_inters=-1, max_user_num=-1,
                remove_same_question=False,**kwargs):
    key_cols = {"user_id": str, "skill_id": str, "problem_id": str, "correct": np.int8, "start_timestamp": np.int64,
                "response_ms": np.int64, ORDER_KEY: np.int64}
    sample_data = pd.read_csv(fn, encoding="utf-8", encoding_errors="ignore", nrows=100)
    use_cols = {}
    id_cols = ["user_id", "skill_id", "problem_id"]
    has_problem = False
    for k, v in key_cols.items():
        if k in sample_data.columns:
            use_cols[k] = v
            if k == "problem_id":
                has_problem = True
        else:
            if k in id_cols:
                id_cols.remove(k)
    # assert all(c in sample_data.columns for c in key_cols + [response_col])
    sample_data.to_csv(os.path.join(dataset_dir, "sample.csv"), encoding="utf-8", index=False)
    df = pd.read_csv(fn, usecols=use_cols.keys(), encoding="utf-8", encoding_errors="ignore",
                     low_memory=False)
    for k, v in use_cols.items():
        col_type = sample_data.dtypes[k]
        if "float" in col_type.name:
            if v == str:
                df[k] = df[k].apply(float2str)
            else:
                df[k] = df[k].apply(float2str)
    org_statistics_data = do_statistics(df, os.path.join(dataset_dir, "org_statistics.json"))
    df = df.dropna(subset=use_cols.keys())
    if min_skill_inters > 0:
        df = df[df.groupby("skill_id")["skill_id"].transform('count').ge(min_skill_inters)]
    if has_problem and min_problem_inters > 0:
        df = df[df.groupby("problem_id")["problem_id"].transform('count').ge(min_problem_inters)]
    user_count = df["user_id"].value_counts()
    min_len_max_user_num = -1
    if 0 < max_user_num < user_count.size:
        min_len_max_user_num = user_count.iloc[max_user_num]
    min_len_1 = max(min_len, min_len_max_user_num)
    if min_len_1 > 0:
        df = df[df.groupby("user_id")["user_id"].transform('count').ge(min_len_1)]
    dropna_statistics_data = do_statistics(df, os.path.join(dataset_dir, "statistics.json"))
    df = df.astype(use_cols, copy=False)
    df['user'], user_set = pd.factorize(df["user_id"], sort=True)
    df['skill'], skill_set = pd.factorize(df["skill_id"], sort=True)
    if has_problem:
        df['problem'], problem_set = pd.factorize(df["problem_id"], sort=True)
    else:
        problem_set = np.array([])
    index_map_data = {"user_map": user_set.tolist(), "skill_map": skill_set.tolist(),
                      "problem_map": problem_set.tolist()}
    with open(os.path.join(dataset_dir, "index_map.json"), "w", encoding="utf-8") as f:
        json.dump(index_map_data, f, indent=4)
    df.drop(columns=id_cols, inplace=True)

    response_statistics_data = do_response_statistics(df, dataset_dir)
    user_num = len(user_set)
    skill_num = len(skill_set)
    problem_num = len(problem_set)
    seq_cols = []
    for k in use_cols.keys():
        if k not in id_cols:
            seq_cols.append(k)
    seq_cols.append('skill')
    if has_problem:
        seq_cols.append('problem')


    def proc_group(r):
        r.sort_values(by=[ORDER_KEY], inplace=True, key=lambda x: pd.to_numeric(x, errors='coerce'))
        if has_problem and remove_same_question:
            p_col = r['problem'].diff(1)
            r=r[p_col != 0.0]
        values = [r[k].values for k in seq_cols]
        values.append(len(r))
        return pd.Series(values, index=seq_cols + ["seq_len"])

    seq = df.groupby('user').apply(
        proc_group
    )
    seq.reset_index(inplace=True)
    # seq = seq[seq["seq_len"] > min_len]
    max_len = seq["seq_len"].max()
    seq.drop(columns=["seq_len"], inplace=True)
    dataset_info = collections.OrderedDict(name=dataset_name,
                                           user_num=int(user_num),
                                           skill_num=int(skill_num),
                                           problem_num=int(problem_num),
                                           org_max_len=int(max_len),
                                           min_len=int(min_len),
                                           min_len_max_user_num=int(min_len_max_user_num),
                                           max_user_num=int(max_user_num),
                                           min_skill_inters=min_skill_inters,
                                           min_problem_inters=min_problem_inters)
    with open(os.path.join(dataset_dir, "info.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=4)
    seq.to_pickle(os.path.join(dataset_dir, "all.pkl"))
    return seq, dataset_info, dataset_dir


def time2ts(t):
    if pd.isna(t):
        return t
    has_float = "." in t
    if has_float:
        time_stamp = datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f").timestamp() * 1000
    else:
        time_stamp = datetime.strptime(t, "%Y-%m-%d %H:%M:%S").timestamp() * 1000
    return int(time_stamp)
