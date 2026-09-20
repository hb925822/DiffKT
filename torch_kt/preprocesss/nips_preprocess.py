#!/usr/bin/env python
# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/6/27 0027 下午 10:42
# @Author  : hb
# @File    : nips_preprocess.py
import os

import pandas as pd

from .utils import parent_dir, to_seq_data, ALL_RAW_FILE_NAME, ORDER_KEY, time2ts, rename_dataset

DATASET_NAME = "nips"
COL_MAPS = {"UserId": "user_id", "SubjectId_level3_str": "skill_id", "QuestionId": "problem_id", "IsCorrect": "correct",
            "answer_timestamp": "start_timestamp"}
SORT_COLS = ["answer_timestamp"]


def process_raw_data(fn, out_dir, sub_data="34", **kwargs):
    """The data downloaded from https://competitions.codalab.org/competitions/25449
    The document can be downloaded from https://arxiv.org/abs/2007.12061.
    """
    if not os.path.exists(fn):
        if sub_data not in ["34", "12"]:
            raise ValueError("task_name only can be 34 or 12")
        task_name = "1_2" if sub_data == "12" else "3_4"
        base_dir = parent_dir(fn)
        meta_data_dir = os.path.join(base_dir, "data", "metadata")
        sequence_data = os.path.join(base_dir, "data", "train_data", "train_task_" + task_name + ".csv")
        print("Start load data")
        answer_metadata_path = os.path.join(meta_data_dir, f"answer_metadata_task_{task_name}.csv")
        question_metadata_path = os.path.join(meta_data_dir, f"question_metadata_task_{task_name}.csv")
        student_metadata_path = os.path.join(meta_data_dir, f"student_metadata_task_{task_name}.csv")
        subject_metadata_path = os.path.join(meta_data_dir, f"subject_metadata.csv")

        df_primary = pd.read_csv(sequence_data)
        print(f"len df_primary is {len(df_primary)}")
        # add timestamp
        df_answer = pd.read_csv(answer_metadata_path)
        df_answer['answer_timestamp'] = df_answer['DateAnswered'].apply(time2ts)
        df_question = pd.read_csv(question_metadata_path)
        # df_student = pd.read_csv(student_metadata_path)
        df_subject = pd.read_csv(subject_metadata_path)

        # only keep level 3
        keep_subject_ids = set(df_subject[df_subject['Level'] == 3]['SubjectId'])
        df_question['SubjectId_level3'] = df_question['SubjectId'].apply(lambda x: set(eval(x)) & keep_subject_ids)

        # merge data
        df_merge = df_primary.merge(df_answer[['AnswerId', 'answer_timestamp']], how='left')  # merge answer time
        df_merge = df_merge.merge(df_question[["QuestionId", "SubjectId_level3"]],
                                  how='left')  # merge question subjects
        df_merge['SubjectId_level3_str'] = df_merge['SubjectId_level3'].apply(lambda x: "_".join([str(i) for i in x]))
        print(f"len df_merge is {len(df_merge)}")
        print("Finish load data")
        print(f"Num of student {df_merge['UserId'].unique().size}")
        print(f"Num of question {df_merge['QuestionId'].unique().size}")
        kcs = []
        for item in df_merge['SubjectId_level3'].values:
            kcs.extend(item)
        print(f"Num of knowledge {len(set(kcs))}")
        print(df_merge.dtypes.to_dict())
        df_merge.to_csv(fn, encoding="utf-8", index=False)
    data_name = rename_dataset(DATASET_NAME, **kwargs) + sub_data
    dataset_dir = os.path.join(out_dir, data_name)
    os.makedirs(dataset_dir, exist_ok=True)
    df = pd.read_csv(fn, encoding="utf-8", encoding_errors="ignore", low_memory=False)
    df.sort_values(by=SORT_COLS, inplace=True, ignore_index=True)
    df[ORDER_KEY] = df.index.values
    df.rename(columns=COL_MAPS, inplace=True)
    fn = os.path.join(dataset_dir, ALL_RAW_FILE_NAME)
    df.to_csv(fn, encoding="utf-8", index=False)
    seq, dataset_info, dataset_dir = to_seq_data(fn, dataset_dir, DATASET_NAME, **kwargs)
    return seq, dataset_info, dataset_dir
