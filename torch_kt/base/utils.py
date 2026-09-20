#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 2024/8/7 11:39
# @Author: hb925
# @File: utils.py
def flatten(nest_data):
    flatted_values = []
    if isinstance(nest_data, (list, set, tuple)):
        for v in nest_data:
            r = flatten(v)
            flatted_values.extend(r)
    elif isinstance(nest_data, dict):
        ks = sorted(nest_data.keys())
        for k in ks:
            r = flatten(nest_data[k])
            flatted_values.extend(r)
    else:
        flatted_values.append(nest_data)
    return flatted_values