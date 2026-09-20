#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 2024/7/20 14:28
# @Author: hb925
# @File: metrics.py
import collections

import sklearn
import numpy as np


# def map_structure(func, *structure, **kwargs):
#   """Creates a new structure by applying `func` to each atom in `structure`.
#
#   Refer to [tf.nest](https://www.tensorflow.org/api_docs/python/tf/nest)
#   for the definition of a structure.
#
#   Applies `func(x[0], x[1], ...)` where x[i] enumerates all atoms in
#   `structure[i]`.  All items in `structure` must have the same arity,
#   and the return value will contain results with the same structure layout.
#
#   Examples:
#
#   * A single Python dict:
#
#   >>> a = {"hello": 24, "world": 76}
#   >>> tf.nest.map_structure(lambda p: p * 2, a)
#   {'hello': 48, 'world': 152}
#
#   * Multiple Python dictionaries:
#
#   >>> d1 = {"hello": 24, "world": 76}
#   >>> d2 = {"hello": 36, "world": 14}
#   >>> tf.nest.map_structure(lambda p1, p2: p1 + p2, d1, d2)
#   {'hello': 60, 'world': 90}
#
#   * A single Python list:
#
#   >>> a = [24, 76, "ab"]
#   >>> tf.nest.map_structure(lambda p: p * 2, a)
#   [48, 152, 'abab']
#
#   * Scalars:
#
#   >>> tf.nest.map_structure(lambda x, y: x + y, 3, 4)
#   7
#
#   * Empty structures:
#
#   >>> tf.nest.map_structure(lambda x: x + 1, ())
#   ()
#
#   * Check the types of iterables:
#
#   >>> s1 = (((1, 2), 3), 4, (5, 6))
#   >>> s1_list = [[[1, 2], 3], 4, [5, 6]]
#   >>> tf.nest.map_structure(lambda x, y: None, s1, s1_list)
#   Traceback (most recent call last):
#   ...
#   TypeError: The two structures don't have the same nested structure
#
#   * Type check is set to False:
#
#   >>> s1 = (((1, 2), 3), 4, (5, 6))
#   >>> s1_list = [[[1, 2], 3], 4, [5, 6]]
#   >>> tf.nest.map_structure(lambda x, y: None, s1, s1_list, check_types=False)
#   (((None, None), None), None, (None, None))
#
#   Args:
#     func: A callable that accepts as many arguments as there are structures.
#     *structure: atom or nested structure.
#     **kwargs: Valid keyword args are:
#       * `check_types`: If set to `True` (default) the types of iterables within
#         the structures have to be same (e.g. `map_structure(func, [1], (1,))`
#         raises a `TypeError` exception). To allow this set this argument to
#         `False`. Note that namedtuples with identical name and fields are always
#         considered to have the same shallow structure.
#       * `expand_composites`: If set to `True`, then composite tensors such as
#         `tf.sparse.SparseTensor` and `tf.RaggedTensor` are expanded into their
#         component tensors.  If `False` (the default), then composite tensors are
#         not expanded.
#
#   Returns:
#     A new structure with the same arity as `structure[0]`, whose atoms
#     correspond to `func(x[0], x[1], ...)` where `x[i]` is the atom in the
#     corresponding location in `structure[i]`. If there are different structure
#     types and `check_types` is `False` the structure types of the first
#     structure will be used.
#
#   Raises:
#     TypeError: If `func` is not callable or if the structures do not match
#       each other by depth tree.
#     ValueError: If no structure is provided or if the structures do not match
#       each other by type.
#     ValueError: If wrong keyword arguments are provided.
#   """
#   if not callable(func):
#     raise TypeError("func must be callable, got: %s" % func)
#
#   if not structure:
#     raise ValueError("Must provide at least one structure")
#
#   check_types = kwargs.pop("check_types", True)
#   expand_composites = kwargs.pop("expand_composites", False)
#
#   if kwargs:
#     raise ValueError(
#         "Only valid keyword arguments are `check_types` and "
#         "`expand_composites`, not: `%s`" % ("`, `".join(kwargs.keys())))
#
#   for other in structure[1:]:
#     assert_same_structure(structure[0], other, check_types=check_types,
#                           expand_composites=expand_composites)
#
#   flat_structure = (flatten(s, expand_composites) for s in structure)
#   entries = zip(*flat_structure)
#
#   return pack_sequence_as(
#       structure[0], [func(*x) for x in entries],
#       expand_composites=expand_composites)
# def assert_same_structure(nest1, nest2, check_types=True,
#                           expand_composites=False):
#     assert is_nested(nest1)
#     assert is_nested(nest2)
#     if check_types:
#         assert type(nest1)==type(nest2)
#     assert len(nest1) == len(nest2)
#     if isinstance(nest1,(list,tuple)):
#         for v1,v2 in zip(nest1,nest2):
#             assert_same_structure(v1,v2)
#     elif isinstance(nest1,dict):
#         if isinstance(nest2,dict):
#             assert set(nest1.keys())==set(nest2.keys())
#             for k,v in nest1.items():
#                 assert_same_structure(v,nest2[k])
#         else:
#             ks=sorted(nest1.keys())
#             for i,k in enumerate(ks):
#                 assert_same_structure(nest1[k],nest2[i])
#     else:
#         pass
def is_nested(nest_data):
    if isinstance(nest_data, (list, set, tuple)):
        for v in nest_data:
            if is_nested(v):
                return True
        return True
    elif isinstance(nest_data, dict):
        for k, v in nest_data.items():
            if is_nested(v):
                return True
        return True
    else:
        return False


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

class Loss(object):
    def __init__(self, name, loss_fn):
        self.name = name
        self.all_loss = []
        self.loss_fn = loss_fn
        self.metric_object = LossMetric(name)

    def __call__(self, y_pred, y, sample_weight=None):
        loss = self.loss_fn(y_pred, y, sample_weight)
        self.metric_object.update_state(loss.detach().cpu().item())
        return loss

    @property
    def metric(self):
        return self.metric_object
# class MeanMetric(Module):
#     def __init__(self, name=None):
#         super().__init__()
#         self.name = name
#         self.total = Parameter(torch.tensor(0, dtype=torch.float), requires_grad=False)
#         self.num = Parameter(torch.tensor(0, dtype=torch.float), requires_grad=False)
#
#     def forward(self, y_true, y_pred, weight=None):
#         pass
#
#     def update(self, v, weight=None):
#         self.total.add_(v.sum())
#         self.num.add_(float(v.numel()))
#
#     def result(self):
#         return self.total / self.num
#
#     def reset(self):
#         self.total.zero_()
#         self.total.zero_()
class MetricsContainer(object):
    def __init__(self, metrics):
        assert not isinstance(metrics, (tuple, list, dict))
        self._metrics = metrics
        self._weighted_metrics = []
        self._metrics_in_order = []

    @property
    def metrics(self):
        """All metrics in this container."""
        return self._metrics_in_order

    def _create_ordered_metrics(self):
        """Cache the flat order needed when return metrics, for backcompat."""
        self._metrics_in_order = []
        for output_metrics, output_weighted_metrics in zip(
                self._metrics, self._weighted_metrics
        ):
            for m in flatten(output_metrics):
                if m is not None:
                    self._metrics_in_order.append(m)
            for wm in flatten(output_weighted_metrics):
                if wm is not None:
                    self._metrics_in_order.append(wm)

    def update_state(self, y_true, y_pred, sample_weight=None):

        zip_args = (
            y_true,
            y_pred,
            sample_weight,
            self._metrics,
        )
        for y_t, y_p, sw, metric_objs in zip(*zip_args):
            # Ok to have no metrics for an output.
            if y_t is None or (
                    all(m is None for m in metric_objs)
            ):
                continue

            # y_t, y_p, sw = match_dtype_and_rank(y_t, y_p, sw)
            # mask = losses_utils.get_mask(y_p)
            # sw = losses_utils.apply_mask(y_p, sw, mask)

            for metric_obj in metric_objs:
                if metric_obj is None:
                    continue
                metric_obj.update_state(y_t, y_p, sample_weight=sw)

    def reste_state(self):
        metrics = self._metrics_in_order
        for metric_obj in metrics:
            if isinstance(metric_obj, Metric):
                metric_obj.reset_state()


class LossContainer(object):
    def __init__(self, losses, output_names,loss_weights):
        assert not isinstance(losses, (tuple, list, dict))
        self._losses = flatten(losses)
        self._loss_weights=flatten(loss_weights)
        self._output_names = flatten(output_names)
        self._per_output_metrics = []
        self._total_loss_mean = LossMetric(name="loss")
        self._create_metrics()

    @property
    def metrics(self):
        per_output_metrics = [
            metric_obj
            for metric_obj in self._per_output_metrics
            if metric_obj is not None
        ]
        return [self._total_loss_mean] + per_output_metrics

    def _create_metrics(self):
        """Creates per-output loss metrics, but only for multi-output Models."""
        if len(self._output_names) == 1:
            self._per_output_metrics = [None]
        else:
            self._per_output_metrics = []
            for loss_obj, output_name in zip(self._losses, self._output_names):
                if loss_obj is None:
                    self._per_output_metrics.append(None)
                else:
                    self._per_output_metrics.append(
                        LossMetric(output_name + "_loss")
                    )

    def __call__(self, y_true, y_pred, sample_weight=None, regularization_losses=None):
        if is_nested(y_pred) and any(is_nested(o) for o in y_pred):
            raise ValueError("do not support this struct of y_pred ")
        y_true, y_pred = flatten(y_true), flatten(y_pred)
        if sample_weight is None:
            sample_weight=[None for _ in y_true]
        else:
            sample_weight = flatten(sample_weight)
        loss_values = []  # Used for gradient calculation.
        total_loss_mean_values = []  # Used for loss metric calculation.
        batch_dim = None
        zip_args = (
            y_true,
            y_pred,
            sample_weight,
            self._losses,
            self._loss_weights,
            self._per_output_metrics,
        )
        for y_t, y_p, sw, loss_obj, loss_weight, metric_obj in zip(*zip_args):
            if (
                    y_t is None or loss_obj is None
            ):  # Ok to have no loss for an output.
                continue
            loss_value = loss_obj(y_t, y_p, sample_weight=sw)

            total_loss_mean_value = loss_value
            # Correct for the `Mean` loss metrics counting each replica as a
            # batch.
            if metric_obj is not None:
                metric_obj.update_state(
                    total_loss_mean_value, sample_weight=batch_dim
                )

            if loss_weight is not None:
                loss_value *= loss_weight
                total_loss_mean_value *= loss_weight

            if (
                    loss_obj.reduction
                    == losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE
                    or loss_obj.reduction == losses_utils.ReductionV2.AUTO
            ):
                loss_value = losses_utils.scale_loss_for_distribution(
                    loss_value
                )

            loss_values.append(loss_value)
            total_loss_mean_values.append(total_loss_mean_value)

        # if regularization_losses:
        #     reg_loss = tf.add_n(regularization_losses)
        #     total_loss_mean_values.append(reg_loss)
        #     loss_values.append(
        #         losses_utils.scale_loss_for_distribution(reg_loss)
        #     )

        if loss_values:
            total_loss_mean_values = losses_utils.cast_losses_to_common_dtype(
                total_loss_mean_values
            )
            total_total_loss_mean_value = tf.add_n(total_loss_mean_values)
            self._total_loss_mean.update_state(
                total_total_loss_mean_value, sample_weight=batch_dim
            )

            loss_values = losses_utils.cast_losses_to_common_dtype(loss_values)
            total_loss = tf.add_n(loss_values)
            return total_loss
        else:
            return None


class Metric(object):
    def __init__(self, name=None):
        super().__init__()
        self.name = name

    def __call__(self, *args, **kwargs):
        self.update_state(*args, **kwargs)

    def update_state(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented in subclasses.")

    def result(self):
        raise NotImplementedError("Must be implemented in subclasses.")

    def reset_state(self):
        raise NotImplementedError("Must be implemented in subclasses.")


class MeanMetricWarpper(Metric):
    def __init__(self, fn, name):
        super().__init__(name)
        self.cal_fn = fn
        self.y_trues = []
        self.y_preds = []

    def update_state(self, y_true, y_pred, weight=None):
        self.y_trues.append(y_true)
        self.y_preds.append(y_pred)

    def result(self):
        ts = np.concatenate(self.y_trues, axis=0)
        ps = np.concatenate(self.y_preds, axis=0)
        v = self.cal_fn(ts, ps)
        return v

    def reset_state(self):
        super().reset_state()
        self.y_trues = []
        self.y_preds = []


class BinaryAcc(MeanMetricWarpper):
    def __init__(self):
        super().__init__(fn=sklearn.metrics.accuracy_score, name="acc")

    def update_state(self, y_true, y_pred, weight=None):
        prelabels = [1 if p >= 0.5 else 0 for p in y_pred]
        super().update_state(y_true, prelabels)


class BinaryAuc(MeanMetricWarpper):
    def __init__(self):
        super().__init__(fn=sklearn.metrics.roc_auc_score, name="auc")


class LossMetric(Metric):
    def __init__(self, name):
        super().__init__(name)
        self.all_loss = []

    def update_state(self, v, weight=None):
        self.all_loss.append(v)

    def result(self):
        return np.mean(self.all_loss)

    def reset_state(self):
        self.all_loss.clear()


if __name__ == "__main__":
    dict0 = {"key3": "value3", "key1": "value1", "key2": "value2"}
    setv = set([1, 3, 4, 5])
    tuple1 = ((1.0, 2.0), (3.0, 4.0, 5.0), 6.0)
    dict1 = {"key3": {"c": (1.0, 2.0), "a": (3.0)}, "key1": {"m": "val1", "g": "val2"}}
    print(flatten(dict0))
    print(flatten(setv))
    print(flatten(tuple1))
    print(flatten(dict1))
