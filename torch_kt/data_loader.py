# _*_ coding: utf-8 _*_
# @Time : 2023/3/1 16:09
# @Author : hb
# @File : data_loader.py
# @desc :
import json
import os
import re

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class DirDataset(object):

    def __init__(self, data_dir):
        self.data_dir = os.path.abspath(data_dir)
        self.info_file = os.path.join(self.data_dir, "info.json")
        assert os.path.exists(self.info_file)
        self.data_file = "all.pkl"
        self.train_file_name = "train.pkl"
        self.val_file_name = "valid.pkl"
        self.test_file_name = "test.pkl"
        self.esp_value = 1e-4
        self.feature_dir = os.path.join(self.data_dir, "features")
        self._info_dict = {}
        self.skill_map_data = {}
        self.problem_map_data = {}
        self.load_info()
        self.load_global_feature()

        if not os.path.exists(self.feature_dir):
            os.makedirs(self.feature_dir, exist_ok=True)

    def load_global_feature(self):
        with open(os.path.join(self.data_dir, "skill_response.json")) as f:
            skill_response = json.load(f)
            skill_response_df = pd.DataFrame(skill_response)
            skill_response_accuracy = skill_response_df["correct"] / skill_response_df["count"]
            skill_response_df["difficulty "] = 1.0-skill_response_accuracy.clip(self.esp_value, 1 - self.esp_value)
            skill_response_df["interact_freq"] = skill_response_df["count"] / skill_response_df["count"].sum()
            skill_response_df["interact_freq"] = skill_response_df["interact_freq"].clip(self.esp_value, 1 - self.esp_value)
            self.skill_map_data["skill_diff"] = skill_response_df.set_index("skill")["difficulty "].to_dict()
            self.skill_map_data["skill_interact_freq"] = skill_response_df.set_index("skill")["interact_freq"].to_dict()
        with open(os.path.join(self.data_dir, "problem_response.json")) as f:
            problem_response = json.load(f)
            problem_response_df = pd.DataFrame(problem_response)
            problem_response_accuracy= problem_response_df["correct"] / problem_response_df["count"]
            problem_response_df["difficulty "] = 1.0-problem_response_accuracy.clip(self.esp_value, 1 - self.esp_value)
            problem_response_df["interact_freq"] = problem_response_df["count"] / problem_response_df["count"].sum()
            problem_response_df["interact_freq"] = problem_response_df["interact_freq"].clip(self.esp_value,
                                                                                         1 - self.esp_value)
            self.problem_map_data["problem_diff"] = problem_response_df.set_index("problem")["difficulty "].to_dict()
            self.problem_map_data["problem_interact_freq"] = problem_response_df.set_index("problem")[
                "interact_freq"].to_dict()

    def _load_or_build_level_mapping(self, key: str, build_func):
        """
        key: e.g., "problem_diff_level_5"
        build_func: callable that returns dict {id: level}
        """
        cache_path = os.path.join(self.feature_dir, f"{key}.json")

        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                mapping = json.load(f)
            # JSON 的 key 是字符串，转回 int（如果 ID 是 int）
            return {int(k): v for k, v in mapping.items()}
        else:
            print(f"🔄 Building and caching {key}...")
            mapping = build_func()
            # 保存时 key 必须是 str（JSON 要求）
            with open(cache_path, 'w') as f:
                json.dump({str(k): int(v) for k, v in mapping.items()}, f)
            return mapping

    def _expand_for_model(self, feature_names):
        for feat in feature_names:
            # --- 试题难度：problem_diff_level_N（等宽）---
            if match := re.match(r"problem_diff_level_(\d+)", feat):
                level_bins = int(match.group(1))
                if f"problem_diff_level_{level_bins}" not in self.problem_map_data:
                    self._build_problem_uniform_level(level_bins)

            # --- 试题难度：problem_diff_freq_level_N（等频）---
            elif match := re.match(r"problem_diff_freq_level_(\d+)", feat):
                level_bins = int(match.group(1))
                if f"problem_diff_freq_level_{level_bins}" not in self.problem_map_data:
                    self._build_problem_quantile_level(level_bins)

            # --- 知识点难度：skill_diff_level_N（等宽）---
            elif match := re.match(r"skill_diff_level_(\d+)", feat):
                level_bins = int(match.group(1))
                if f"skill_diff_level_{level_bins}" not in self.skill_map_data:
                    self._build_skill_uniform_level(level_bins)

            # --- 知识点难度：skill_diff_freq_level_N（等频）---
            elif match := re.match(r"skill_diff_freq_level_(\d+)", feat):
                level_bins = int(match.group(1))
                if f"skill_diff_freq_level_{level_bins}" not in self.skill_map_data:
                    self._build_skill_quantile_level(level_bins)

    def _build_problem_uniform_level(self, level_bins):
        def build():
            base_map = self.problem_map_data["problem_diff"]
            return {
                pid: min(int(diff * level_bins), level_bins - 1)
                for pid, diff in base_map.items()
            }

        key = f"problem_diff_level_{level_bins}"
        self.problem_map_data[key] = self._load_or_build_level_mapping(key, build)

    def _build_problem_quantile_level(self, level_bins):
        def build():
            base_map = self.problem_map_data["problem_diff"]
            all_diffs = np.array(list(base_map.values()))
            if all_diffs.min() == all_diffs.max():
                raise ValueError(
                    f"Quantile binning (v-level) requires variation in difficulty. "
                    f"All {len(all_diffs)} problems have the same difficulty ({all_diffs[0]:.6f}). "
                    f"Consider using uniform binning (e.g., 'problem_diff_level_{level_bins}') instead."
                )
            boundaries = np.quantile(all_diffs, np.linspace(0, 1, level_bins + 1)[1:-1])
            return {
                pid: int(np.searchsorted(boundaries, diff, side='right'))
                for pid, diff in base_map.items()
            }

        key = f"problem_diff_freq_level_{level_bins}"
        self.problem_map_data[key] = self._load_or_build_level_mapping(key, build)

    def _build_skill_uniform_level(self, level_bins):
        def build():
            base_map = self.skill_map_data["skill_diff"]
            return {
                sid: min(int(diff * level_bins), level_bins - 1)
                for sid, diff in base_map.items()
            }

        key = f"skill_diff_level_{level_bins}"
        self.skill_map_data[key] = self._load_or_build_level_mapping(key, build)

    def _build_skill_quantile_level(self, level_bins):
        def build():
            base_map = self.skill_map_data["skill_diff"]
            all_diffs = np.array(list(base_map.values()))
            if all_diffs.min() == all_diffs.max():
                raise ValueError(
                    f"Quantile binning (v-level) requires variation in difficulty. "
                    f"All {len(all_diffs)} skills have the same difficulty ({all_diffs[0]:.6f}). "
                    f"Consider using uniform binning (e.g., 'problem_diff_level_{level_bins}') instead."
                )
            boundaries = np.quantile(all_diffs, np.linspace(0, 1, level_bins + 1)[1:-1])
            return {
                sid: int(np.searchsorted(boundaries, diff, side='right'))
                for sid, diff in base_map.items()
            }

        key = f"skill_diff_freq_level_{level_bins}"
        self.skill_map_data[key] = self._load_or_build_level_mapping(key, build)

    def load_info(self):
        with open(self.info_file, "r", encoding="utf-8") as f:
            self._info_dict.update(json.load(f))

    def calc_max_len(self):
        data_paths = [os.path.join(self.data_dir, "train.pkl"),
                      os.path.join(self.data_dir, "valid.pkl"),
                      os.path.join(self.data_dir, "test.pkl")]
        if all([os.path.exists(f) for f in data_paths]):
            all_datas = pd.concat([pd.read_pickle(f) for f in data_paths])
        else:
            all_datas = pd.read_pickle(os.path.join(self.data_dir, "all.pkl"))
        data_max_len = all_datas["correct"].apply(lambda x: len(x)).max()
        return data_max_len

    @property
    def params(self):
        return dict(data_name=self.name, skill_num=self.skill_num,
                    problem_num=self.problem_num)

    @property
    def data_info(self):
        return self._info_dict

    @property
    def name(self):
        return self._info_dict.get("name", "unknow")

    @property
    def max_len(self):
        return self.calc_max_len()
        # v = self._info_dict.get("max_len", -1)
        # if v == -1:
        #     max_len = self.calc_max_len()
        #     self._info_dict.update({"max_len": max_len})
        #     return max_len
        # else:
        #     return v

    @property
    def org_max_len(self):
        return self._info_dict.get("org_max_len", -1)

    @property
    def skill_num(self):
        return self._info_dict.get("skill_num", -1)

    @property
    def problem_num(self):
        return self._info_dict.get("problem_num", -1)

    @property
    def user_num(self):
        return self._info_dict.get("user_num", -1)

    def filter_datas(self, max_len, data_file_dir=None, drop_remains=False):
        if data_file_dir is None:
            data_file_dir = self.data_dir
        data_paths = [os.path.join(data_file_dir, self.train_file_name),
                      os.path.join(data_file_dir, self.val_file_name),
                      os.path.join(data_file_dir, self.test_file_name)]
        if any([not os.path.exists(f) for f in data_paths]):
            self.expand_split_data(out_dir=data_file_dir)
        if max_len > 0:
            new_paths = [f.replace(".pkl", f'-{max_len}.pkl') for f in data_paths]
            for f, f1 in zip(data_paths, new_paths):
                if not os.path.exists(f1):
                    data = pd.read_pickle(f)
                    new_data = self.filter_maxlength(data, max_len, drop_remains=drop_remains)
                    pd.to_pickle(new_data, f1)
        else:
            new_paths = data_paths
        return new_paths

    def recheck_data(self, folds=-1, max_len=-1, drop_remains=False):
        self.filter_datas(max_len)
        if folds > 0:
            sub_data_dir = [os.path.join(self.data_dir, f'k{i}') for i in range(folds)]
            if any([not os.path.exists(f) for f in sub_data_dir]):
                self.expand_kf_data(max_len=-1)
            for d in sub_data_dir:
                self.filter_datas(max_len, d)

    def expand_kf_data(self, target_dir=None, max_len=-1, min_len=3, drop_remains=False):
        print("############### expand kf data ################")
        if target_dir is None:
            target_dir = self.data_dir
        all_file_path = os.path.join(self.data_dir, "all.pkl")
        info_file_path = os.path.join(self.data_dir, "info.json")
        assert os.path.exists(all_file_path)
        assert os.path.exists(info_file_path)
        all_datas = pd.read_pickle(all_file_path)
        assert "fold" in all_datas.columns
        train_valid_data = all_datas[all_datas["fold"] != -1]
        test_data = all_datas[all_datas["fold"] == -1]
        if len(test_data) <= 0:
            test_data = None
        with open(info_file_path, "r", encoding="utf-8") as f:
            dataset_info = json.load(f)
            folds = train_valid_data["fold"].astype(np.int32).value_counts().keys()
            for k in folds:
                fold_filter = train_valid_data["fold"] == k
                valid_data = train_valid_data[fold_filter].reset_index(drop=True)
                train_data = train_valid_data[~fold_filter].reset_index(drop=True)
                if test_data is None:
                    test_data = valid_data
                print(f'train:{len(train_data)},valid:{len(valid_data)},test:{len(test_data)}')
                sub_out_dir = os.path.join(target_dir, f'k{k}')
                os.makedirs(sub_out_dir, exist_ok=True)
                dataset_info_sub = dataset_info.copy()
                if max_len and max_len > 0:
                    train_data = self.filter_maxlength(train_data, max_len, min_len, drop_remains)
                    valid_data = self.filter_maxlength(valid_data, max_len, min_len, drop_remains)
                    test_data = self.filter_maxlength(test_data, max_len, min_len, drop_remains)
                self.save_dataset(train_data, valid_data, test_data, dataset_info_sub, sub_out_dir)
        print("############### expand kf data over################")

    @staticmethod
    def save_dataset(train_data, valid_data, test_data, data_info, out_dir, all_datas=None, ext_flag=""):
        os.makedirs(out_dir, exist_ok=True)
        if train_data is not None:
            train_data.to_pickle(os.path.join(out_dir, f"train{ext_flag}.pkl"))
        if valid_data is not None:
            valid_data.to_pickle(os.path.join(out_dir, f"valid{ext_flag}.pkl"))
        if test_data is not None:
            test_data.to_pickle(os.path.join(out_dir, f"test{ext_flag}.pkl"))
        if all_datas is not None:
            all_datas.to_pickle(os.path.join(out_dir, "all.pkl"))
        with open(os.path.join(out_dir, "info.json"), "w", encoding="utf-8") as f:
            json.dump(data_info, f, indent=4)

    def split_generator(self, data_file_dir=None, feature_names=None, label_names=None, sample_num=-1, max_len=-1,
                        drop_remains=False):
        self._expand_for_model(feature_names)
        if data_file_dir is None:
            data_file_dir = self.data_dir
        if max_len > 0:
            data_paths = self.filter_datas(max_len, data_file_dir, drop_remains)
        else:
            data_paths = [os.path.join(data_file_dir, self.train_file_name),
                          os.path.join(data_file_dir, self.val_file_name),
                          os.path.join(data_file_dir, self.test_file_name)]
        datasets = [KtDataSet(f, self.skill_num, self.problem_num, feature_names, label_names,
                              samples=sample_num, problem_map_data=self.problem_map_data,
                              skill_map_data=self.skill_map_data) for
                    f in
                    data_paths]
        if max_len > 0:
            datasets = [d.padding(max_len) for d in datasets]
        return datasets

    def expand_split_data(self, max_len=-1, min_len=3, out_dir=None):
        all_file_path = os.path.join(self.data_dir, "all.pkl")
        info_file_path = os.path.join(self.data_dir, "info.json")
        if out_dir is None or not os.path.exists(out_dir):
            out_dir = self.data_dir
        assert os.path.exists(all_file_path)
        assert os.path.exists(info_file_path)
        all_datas = pd.read_pickle(all_file_path)
        assert "split" in all_datas.columns
        with open(info_file_path, "r", encoding="utf-8") as f:
            dataset_info = json.load(f)
            train_data = all_datas[all_datas["split"] == 0].reset_index(drop=True)
            valid_data = all_datas[all_datas["split"] == 1].reset_index(drop=True)
            test_data = all_datas[all_datas["split"] == -1].reset_index(drop=True)
            if max_len and max_len > 0:
                train_data = self.filter_maxlength(train_data, max_len, min_len)
                valid_data = self.filter_maxlength(valid_data, max_len, min_len)
                test_data = self.filter_maxlength(test_data, max_len, min_len)
            if train_data is not None:
                train_data.to_pickle(os.path.join(out_dir, "train.pkl"))
            if valid_data is not None:
                valid_data.to_pickle(os.path.join(out_dir, "valid.pkl"))
            if test_data is not None:
                test_data.to_pickle(os.path.join(out_dir, "test.pkl"))
            with open(os.path.join(out_dir, "info.json"), "w", encoding="utf-8") as f:
                json.dump(dataset_info, f, indent=4)
            self._info_dict.update(dataset_info)

    @classmethod
    def filter_maxlength(cls, all_data, max_len, min_len=3, drop_remains=False):
        assert {'user', "correct"} < set(all_data.columns)
        all_data["seq_len"] = all_data["correct"].apply(lambda x: len(x))

        def sub_seq(r):
            s_len = r["seq_len"]
            datas = []
            if s_len > max_len:
                if drop_remains:
                    v = 1
                else:
                    v = s_len // max_len
                    v = v if (s_len % max_len) < min_len else v + 1
                for i in range(v):
                    j = i * max_len
                    d = {}
                    for ind in r.index:
                        if not isinstance(r[ind], (list, tuple, np.ndarray)):
                            d[ind] = r[ind]
                        else:
                            d[ind] = r[ind][j:j + max_len]
                        if ind == "seq_len":
                            d[ind] = max_len
                    datas.append(d)
                return pd.DataFrame(datas, columns=r.index)
            else:
                return pd.DataFrame([r])

        filter_index = all_data["seq_len"] > max_len
        tmp_data = all_data[filter_index].apply(sub_seq, axis=1)
        seq = pd.concat(tmp_data.to_list() + [all_data[~filter_index]]).reset_index(drop=True)
        seq.drop(columns=["seq_len"], inplace=True)
        # dataset_info["name"] = f'{dataset_info["name"]}_l{max_len}'
        return seq


class KtDataSet(Dataset):
    def __init__(self, pklfile_or_dataframe, skill_num, problem_num, feature_names=None, label_names=None,
                 samples=-1, problem_map_data=None, skill_map_data=None, **kwargs):
        if feature_names is None:
            feature_names = ["skill", "skill_response", "problem_response", "problem"]
        if label_names is None:
            label_names = ["correct"]
        if isinstance(pklfile_or_dataframe, (str,)) and pklfile_or_dataframe.endswith(".pkl"):
            df = pd.read_pickle(pklfile_or_dataframe)
        else:
            df = pklfile_or_dataframe
        if samples > 0:
            if samples >= 1:
                df = df.sample(int(samples))
            else:
                df = df.sample(frac=samples)
        if isinstance(label_names, str):
            label_names = [label_names]
        else:
            label_names = list(label_names)
        if isinstance(feature_names, str):
            feature_names = [feature_names]
        else:
            feature_names = list(feature_names)

        self.feature_names = feature_names
        self.label_names = label_names
        self.seq = df
        self.skill_num = skill_num
        self.problem_num = problem_num
        self.position = -1
        self.size = len(self.seq)
        self.skill_map_data = skill_map_data
        self.problem_map_data = problem_map_data
        self._init_data()

    def padding(self, max_len, pad_in_end=True):
        columns = set(self.seq.columns)
        for column in columns:
            if pad_in_end:
                self.seq[column] = self.seq[column].apply(
                    lambda x: np.pad(x, (0, max_len - len(x)), constant_values=-1) if isinstance(x, np.ndarray) else x)
            else:
                self.seq[column] = self.seq[column].apply(
                    lambda x: np.pad(x, (max_len - len(x), 0), constant_values=-1) if isinstance(x, np.ndarray) else x)
        return self

    def _init_data(self):
        for k in set(self.feature_names + self.label_names) - set(self.seq.columns):
            if k in self.skill_map_data.keys():
                values = self.skill_map_data[k]
                self.seq[k] = self.seq["skill"].apply(lambda x: np.array([values[i] for i in x]))
            if k in self.problem_map_data.keys():
                values = self.problem_map_data[k]
                self.seq[k] = self.seq["problem"].apply(lambda x: np.array([values[i] for i in x]))
            if k == "skill_response":
                self.seq[k] = self.seq[["skill", "correct"]].apply(lambda x: x["skill"] + x["correct"].astype(int) * self.skill_num,
                                                                   axis=1)
            if k == "problem_response":
                self.seq[k] = self.seq[["problem", "correct"]].apply(
                    lambda x: x["problem"] + x["correct"].astype(int) * self.problem_num, axis=1)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        row = self.seq.iloc[index]
        self.position += 1
        x_datas = [row[arg] for arg in self.feature_names]
        y_datas = [row[arg] for arg in self.label_names]
        if len(x_datas) > 1:
            x_datas = tuple(x_datas)
        else:
            x_datas = x_datas[0]
        if len(y_datas) > 1:
            y_datas = tuple(y_datas)
        else:
            y_datas = y_datas[0]
        return x_datas, y_datas
