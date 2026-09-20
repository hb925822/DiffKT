#!/usr/bin/env python
# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/6/17 0017 上午 9:19
# @Author  : hb
# @File    : wandb_utils.py
import json
import logging
import os
import shutil
import sys
import traceback

import pandas as pd
import wandb
from wandb.sdk.artifacts.exceptions import ArtifactNotLoggedError


def get_wandb_config(config_file):
    with open(config_file, "r", encoding="utf-8") as f:
        wandb_co = json.load(f)
        os.environ["WANDB_PROJECT"] = wandb_co["project"]
        os.environ["WANDB_ENTITY"] = wandb_co["entity"]
        os.environ["WANDB_API_KEY"] = wandb_co["wandb_key"]
        os.environ['WANDB__EXECUTABLE'] = sys.executable
        wandb.login(key=os.environ["WANDB_API_KEY"], host=wandb_co["wandb_host"])
    return wandb_co["wandb_host"], wandb_co["wandb_key"], wandb_co["project"], wandb_co["entity"]


def download_wandb_dataset(data_base, data_name, alias="all", is_force=False):
    dataset_dir = os.path.join(data_base, data_name)
    dataset_tmp_dir = os.path.join(data_base, "cache", data_name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)
    if not os.path.exists(dataset_tmp_dir):
        os.makedirs(dataset_tmp_dir, exist_ok=True)
    try:
        data_artifact = wandb.Api().artifact(f"{data_name}:{alias}", type="DATASET")
        if is_force:
            shutil.rmtree(dataset_dir, ignore_errors=True)
            shutil.rmtree(dataset_tmp_dir, ignore_errors=True)
        try:
            print("verifing the directory...")
            data_artifact.verify(dataset_tmp_dir)
        except Exception as e:
            shutil.rmtree(dataset_dir, ignore_errors=True)
            shutil.rmtree(dataset_tmp_dir, ignore_errors=True)
            os.makedirs(dataset_tmp_dir, exist_ok=True)
            os.makedirs(dataset_dir, exist_ok=True)
            print("verify faild,downloading dataname...")
            data_artifact.checkout(dataset_tmp_dir)
            shutil.copytree(dataset_tmp_dir, dataset_dir, dirs_exist_ok=True)
    except Exception as e:
        raise ValueError("data_artifact error")
    return data_artifact,dataset_dir


def log_to_wandb(wandb_run, test_result, run_log_dir, model_name, dataset_name):
    model_artifact_name = model_name + '_' + dataset_name
    wandb_run.summary.update(test_result)
    # wandb.Settings(code_dir=".")
    wandb_run.log_code("../", include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"))
    ma = make_model_atifact(run_log_dir, model_artifact_name)
    wandb_run.log_artifact(ma)
    test_result_copy = test_result.copy()
    test_result_copy.update({"model": model_name, "dataname": dataset_name})
    my_table = wandb.Table(dataframe=pd.DataFrame([test_result_copy]))
    wandb_run.log({"EvalResult": my_table})
    wandb_run.finish()


def sync_tensorboard_history(d, root_path, data_root_path, api, en_debug=True, entity="hebo",
                             project="knowledge_trace", job_type="modle_test", max_epoch=50, batch_size=32,
                             optimizer="adam"):
    def check_logs(log_dir):
        subfiles = ["test_result.csv", "train.csv", "train", "validation", "weights"]
        if not set(subfiles) <= set(os.listdir(log_dir)):
            raise ValueError("日志不完整!!!")
        if not (os.listdir(os.path.join(log_dir, "train")) and os.listdir(os.path.join(log_dir, "validation"))):
            raise ValueError("日志不完整!!!")
    try:
        if en_debug:
            logger = logging.getLogger("wandb")
            logger.setLevel(logging.DEBUG)
        i0 = d.find("_")
        i1 = d.rfind("_")
        m_name = d[:i0]
        d_name = d[i0 + 1:i1]
        time_str = d[i1 + 1:]
        run_id = None
        run = None
        check_logs(os.path.join(root_path, d))
        exists_runs = api.runs(
            path=f"{entity}/{project}",
            filters={"display_name": m_name, "group": d_name, "jobType": "modle_test"}
        )
        try:
            if len(exists_runs) > 0:
                run_id = exists_runs[0].id
        except:
            pass
        # run1 = api.sync_tensorboard(os.path.join(root_path, d), run_id, entity=entity, project=project)
        # run1.wait_until_finished()
        config = {"dataname": d_name, "max_epoch": max_epoch, "batch_size": batch_size, "optimizer": optimizer}
        # run1.name=m_name
        # run1.group=d_name
        # run1.update()
        # if run1.state != 'finished':
        #     raise ValueError("sync_tensorboard faild")
        run = wandb.init(id=run_id, job_type=job_type, config=config, resume="must")
        try:
            data_artifact = api.artifact(f'{entity}/{project}/{d_name}:latest', type='DATASET')
            run.use_artifact(data_artifact)
        except:
            data_path = os.path.join(data_root_path, d_name)
            da = make_dataset_atifact(data_path)
            run.log_artifact(da, aliases=['latest'])
        try:
            model_artifact = api.artifact(f'{entity}/{project}/{m_name}_{d_name}:latest', type='MODEL')
            run.use_artifact(model_artifact)
        except:
            ma = wandb.Artifact(m_name + '_' + d_name, type="MODEL")
            ma.add_dir(os.path.join(root_path, d, 'weights'), name="weights")
            if os.path.exists(os.path.join(root_path, d, "model_config.json")):
                ma.add_file(os.path.join(root_path, d, "model_config.json"))
            run.log_artifact(ma, aliases=['latest'])
        test_result_f = os.path.join(root_path, d, 'test_result.csv')
        test_result = pd.read_csv(test_result_f, encoding='utf-8', index_col=None)
        test_result_dict = test_result.iloc[0].to_dict()
        run.summary.update(test_result_dict)
        test_result_dict.update({"model": m_name, "dataname": d_name})
        my_table = wandb.Table(dataframe=pd.DataFrame([test_result_dict]))
        run.log({"EvalResult": my_table})
        run.finish()
    except Exception as e:
        traceback.print_exc()
        if run:
            run.finish(-1)


def make_model_atifact(run_log_dir, model_artifact_name):
    ma = wandb.Artifact(model_artifact_name, type="MODEL")
    if os.path.exists(os.path.join(run_log_dir, "model.png")):
        ma.add_file(os.path.join(run_log_dir, "model.png"), name="model_img.png")
    if os.path.exists(os.path.join(run_log_dir, "predict.csv")):
        ma.add_file(os.path.join(run_log_dir, "predict.csv"), name="predict.csv")
    if os.path.exists(os.path.join(run_log_dir, "train.csv")):
        ma.add_file(os.path.join(run_log_dir, "train.csv"), name="train.csv")
    ma.add_dir(os.path.join(run_log_dir, 'weights'), name="weights")
    return ma


def make_dataset_atifact(dataset_dir, include_all=False):
    try:
        with open(os.path.join(dataset_dir, "info.json"), "r", encoding="utf-8") as f:
            dataset_info = json.load(f)
        meta_data = dataset_info.copy()
        da = wandb.Artifact(dataset_info["name"], type="DATASET", metadata=meta_data)
        sub_files = os.listdir(dataset_dir)
        if include_all:
            da.add_dir(dataset_dir)
        else:
            for subfile in sub_files:
                if subfile in ["info.json", "all.pkl",   "sample.csv",
                               "index_map.json", "statistics.json", "org_statistics.json",
                               "skill_problem_count.json", "problem_skill_count.json",
                               "skill_response.json", "problem_response.json",
                               "skill_problem_response.json"]:
                    da.add_file(os.path.join(dataset_dir, subfile))
        if "statistics.json" in sub_files:
            with open(os.path.join(dataset_dir, "statistics.json"), "r", encoding="utf-8") as f:
                colstat_value = json.load(f)
                t1 = wandb.Table(dataframe=pd.DataFrame(colstat_value["interaction_statistics"]))
                da.add(t1, name="interaction_statistics")
        if "org_statistics.json" in sub_files:
            with open(os.path.join(dataset_dir, "org_statistics.json"), "r", encoding="utf-8") as f:
                org_colstat_value = json.load(f)
                t1 = wandb.Table(dataframe=pd.DataFrame(org_colstat_value["interaction_statistics"]))
                da.add(t1, name="org_" + "interaction_statistics")
    except Exception as e:
        print(e)
    return da


def make_split_dataset_atifact(dataset_dir):
    with open(os.path.join(dataset_dir, "info.json"), "r", encoding="utf-8") as f:
        dataset_info = json.load(f)
    meta_data = dataset_info.copy()
    da = wandb.Artifact(dataset_info["name"], type="DATASET", metadata=meta_data)
    da.add_file(os.path.join(dataset_dir, "info.json"))
    da.add_file(os.path.join(dataset_dir, "test.pkl"))
    da.add_file(os.path.join(dataset_dir, "train.pkl"))
    da.add_file(os.path.join(dataset_dir, "valid.pkl"))
    return da


def make_kf_dataset_atifact(dataset_dir):
    with open(os.path.join(dataset_dir, "info.json"), "r", encoding="utf-8") as f:
        dataset_info = json.load(f)
    meta_data = dataset_info.copy()
    da = wandb.Artifact(dataset_info["name"], type="DATASET", metadata=meta_data)
    da.add_file(os.path.join(dataset_dir, "info.json"))
    da.add_file(os.path.join(dataset_dir, "test.pkl"))
    da.add_file(os.path.join(dataset_dir, "train_valid.pkl"))
    # for k in range(dataset_info["folds"]):
    #     da.add_dir(os.path.join(dataset_dir, f'k{k}'), f'k{k}')
    return da


def login_wandb(wandb_host, wandb_key, project, entity, debug=False):
    os.environ["WANDB_PROJECT"] = project
    os.environ["WANDB_ENTITY"] = entity
    os.environ["WANDB_API_KEY"] = wandb_key
    # os.environ["WANDB_DISABLE_GIT"] = "true"
    if debug:
        logger = logging.getLogger("wandb")
        logger.setLevel(logging.DEBUG)
    wandb.login(key=wandb_key, host=wandb_host)


# def get_dataset_artifact(data_name, folds, data_base):
#     dataset_dir = os.path.join(data_base, data_name)
#     data_artifact = wandb.Api().artifact(f'{data_name}:kf-{folds}',
#                                          type="DATASET")
#     os.makedirs(dataset_dir, exist_ok=True)
#     dir_files = os.listdir(dataset_dir)
#     s = [f'k{i}' in dir_files for i in range(folds)]
#     if not all(s):
#         try:
#             print("verifing the directory...")
#             data_artifact.verify(dataset_dir)
#         except:
#             print("verify faild,downloading dataname...")
#             data_artifact.checkout(dataset_dir)
#         expand_kf_data(dataset_dir)
#     return data_artifact.metadata
