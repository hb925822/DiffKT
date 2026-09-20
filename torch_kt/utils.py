#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 2024/8/29 20:45
# @Author: hb925
# @File: utils.py
import os

PROJECT_NAME="knowledge_trace_torch"


def root_dir():
    current_path = os.path.abspath(__file__)
    index = current_path.find(PROJECT_NAME)
    if index >= 0:
        return current_path[:index + len(PROJECT_NAME)]
    else:
        return os.getcwd()


