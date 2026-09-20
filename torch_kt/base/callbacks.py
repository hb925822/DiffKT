#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 2024/7/20 14:57
# @Author: hb925
# @File: callbacks.py
import collections
import csv
import os
import sys
import time

import numpy as np

from torch_kt.base.utils import flatten


def print_msg(message, line_break=True):
    if line_break:
        sys.stdout.write(message + "\n")
    else:
        sys.stdout.write(message)
    sys.stdout.flush()


class Progbar:
    """Displays a progress bar.

    Args:
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that should *not*
          be averaged over time. Metrics in this list will be displayed as-is.
          All others will be averaged by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
        unit_name: Display name for step counts (usually "step" or "sample").
    """

    def __init__(
            self,
            target,
            width=30,
            verbose=1,
            interval=0.05,
            unit_name="step",
    ):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        self.unit_name = unit_name
        self._dynamic_display = (
                (hasattr(sys.stdout, "isatty") and sys.stdout.isatty())
                or "ipykernel" in sys.modules
                or "posix" in sys.modules
                or "PYCHARM_HOSTED" in os.environ
        )
        self._total_width = 0
        self._seen_so_far = 0
        # We use a dict + list to avoid garbage collection
        # issues found in OrderedDict
        self._values = {}
        self._values_order = []
        self._start = time.time()
        self._last_update = 0
        self._time_at_epoch_start = self._start
        self._time_at_epoch_end = None
        self._time_after_first_step = None

    def update(self, current, values=None, finalize=None):
        """Updates the progress bar.

        Args:
            current: Index of current step.
            values: List of tuples: `(name, value_for_last_step)`. If `name` is
              in `stateful_metrics`, `value_for_last_step` will be displayed
              as-is. Else, an average of the metric over time will be
              displayed.
            finalize: Whether this is the last update for the progress bar. If
              `None`, defaults to `current >= self.target`.
        """
        if finalize is None:
            if self.target is None:
                finalize = False
            else:
                finalize = current >= self.target

        values = values or []
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)
            if k not in self.stateful_metrics:
                # In the case that progress bar doesn't have a target value in
                # the first epoch, both on_batch_end and on_epoch_end will be
                # called, which will cause 'current' and 'self._seen_so_far' to
                # have the same value. Force the minimal value to 1 here,
                # otherwise stateful_metric will be 0s.
                value_base = max(current - self._seen_so_far, 1)
                if k not in self._values:
                    self._values[k] = [v * value_base, value_base]
                else:
                    self._values[k][0] += v * value_base
                    self._values[k][1] += value_base
            else:
                # Stateful metrics output a numeric value. This representation
                # means "take an average from a single value" but keeps the
                # numeric formatting.
                self._values[k] = [v, 1]
        self._seen_so_far = current

        message = ""
        now = time.time()
        info = " - %.0fs" % (now - self._start)
        if current == self.target:
            self._time_at_epoch_end = now
        if self.verbose == 1:
            if now - self._last_update < self.interval and not finalize:
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                message += "\b" * prev_total_width
                message += "\r"
            else:
                message += "\n"

            if self.target is not None:
                numdigits = int(np.log10(self.target)) + 1
                bar = ("%" + str(numdigits) + "d/%d [") % (current, self.target)
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += "=" * (prog_width - 1)
                    if current < self.target:
                        bar += ">"
                    else:
                        bar += "="
                bar += "." * (self.width - prog_width)
                bar += "]"
            else:
                bar = "%7d/Unknown" % current

            self._total_width = len(bar)
            message += bar

            time_per_unit = self._estimate_step_duration(current, now)

            if self.target is None or finalize:
                info += self._format_time(time_per_unit, self.unit_name)
            else:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = "%d:%02d:%02d" % (
                        eta // 3600,
                        (eta % 3600) // 60,
                        eta % 60,
                    )
                elif eta > 60:
                    eta_format = "%d:%02d" % (eta // 60, eta % 60)
                else:
                    eta_format = "%ds" % eta

                info = " - ETA: %s" % eta_format

            for k in self._values_order:
                info += " - %s:" % k
                if isinstance(self._values[k], list):
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1])
                    )
                    if abs(avg) > 1e-3:
                        info += " %.4f" % avg
                    else:
                        info += " %.4e" % avg
                else:
                    info += " %s" % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += " " * (prev_total_width - self._total_width)

            if finalize:
                info += "\n"

            message += info
            print_msg(message, line_break=False)
            message = ""

        elif self.verbose == 2:
            if finalize:
                numdigits = int(np.log10(self.target)) + 1
                count = ("%" + str(numdigits) + "d/%d") % (current, self.target)
                info = count + info
                for k in self._values_order:
                    info += " - %s:" % k
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1])
                    )
                    if avg > 1e-3:
                        info += " %.4f" % avg
                    else:
                        info += " %.4e" % avg
                if self._time_at_epoch_end:
                    time_per_epoch = (
                            self._time_at_epoch_end - self._time_at_epoch_start
                    )
                    avg_time_per_step = time_per_epoch / self.target
                    self._time_at_epoch_start = now
                    self._time_at_epoch_end = None
                    info += " -" + self._format_time(time_per_epoch, "epoch")
                    info += " -" + self._format_time(
                        avg_time_per_step, self.unit_name
                    )
                    info += "\n"
                message += info
                print_msg(message, line_break=False)
                message = ""

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)

    def _format_time(self, time_per_unit, unit_name):
        """format a given duration to display to the user.

        Given the duration, this function formats it in either milliseconds
        or seconds and displays the unit (i.e. ms/step or s/epoch)
        Args:
          time_per_unit: the duration to display
          unit_name: the name of the unit to display
        Returns:
          a string with the correctly formatted duration and units
        """
        formatted = ""
        if time_per_unit >= 1 or time_per_unit == 0:
            formatted += " %.0fs/%s" % (time_per_unit, unit_name)
        elif time_per_unit >= 1e-3:
            formatted += " %.0fms/%s" % (time_per_unit * 1e3, unit_name)
        else:
            formatted += " %.0fus/%s" % (time_per_unit * 1e6, unit_name)
        return formatted

    def _estimate_step_duration(self, current, now):
        """Estimate the duration of a single step.

        Given the step number `current` and the corresponding time `now` this
        function returns an estimate for how long a single step takes. If this
        is called before one step has been completed (i.e. `current == 0`) then
        zero is given as an estimate. The duration estimate ignores the duration
        of the (assumed to be non-representative) first step for estimates when
        more steps are available (i.e. `current>1`).

        Args:
          current: Index of current step.
          now: The current time.

        Returns: Estimate of the duration of a single step.
        """
        if current:
            # there are a few special scenarios here:
            # 1) somebody is calling the progress bar without ever supplying
            #    step 1
            # 2) somebody is calling the progress bar and supplies step one
            #    multiple times, e.g. as part of a finalizing call
            # in these cases, we just fall back to the simple calculation
            if self._time_after_first_step is not None and current > 1:
                time_per_unit = (now - self._time_after_first_step) / (
                        current - 1
                )
            else:
                time_per_unit = (now - self._start) / current

            if current == 1:
                self._time_after_first_step = now
            return time_per_unit
        else:
            return 0

    def _update_stateful_metrics(self, stateful_metrics):
        self.stateful_metrics = self.stateful_metrics.union(stateful_metrics)


class Callback(object):
    def __init__(self):
        self.validation_data = None
        self.model = None
        self.params = None

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def on_batch_begin(self, batch, logs=None):
        '''

        :param batch:
        :param logs:
        :return:
        '''

    def on_batch_end(self, batch, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        pass

    def on_predict_batch_begin(self, batch, logs=None):
        pass

    def on_predict_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_test_begin(self, logs=None):
        pass

    def on_test_end(self, logs=None):
        pass

    def on_predict_begin(self, logs=None):
        pass

    def on_predict_end(self, logs=None):
        pass

class CallbackList(object):
    def __init__(self, callbacks=None,
                 add_history=False,
                 add_progbar=False,
                 model=None, ):
        self.callbacks = callbacks if callbacks else []
        # self._add_default_callbacks(add_history, add_progbar)

    def on_batch_begin(self, batch, logs=None):
        '''

        :param batch:
        :param logs:
        :return:
        '''
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)
    def on_batch_end(self, batch, logs=None):
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def on_epoch_begin(self, epoch, logs=None):
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        pass

    def on_predict_batch_begin(self, batch, logs=None):
        pass

    def on_predict_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_test_begin(self, logs=None):
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_test_begin(logs)

    def on_test_end(self, logs=None):
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_test_end(logs)

    def on_predict_begin(self, logs=None):
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_predict_begin(logs)

    def on_predict_end(self, logs=None):
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_predict_end(logs)

    def _process_logs(self, logs):
        """Turns tensors into numpy arrays or Python scalars if necessary."""
        if logs is None:
            return {}
        assert isinstance(logs, dict)
        return logs

    def _call_batch_hook_helper(self, hook_name, batch, logs):
        """Helper function for `on_*_batch_*` methods."""
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            hook = getattr(callback, hook_name)
            hook(batch, logs)





class CSVLogger(Callback):
    def __init__(self, filename, separator=",", append=False):
        self.sep = separator
        self.filename = filename
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        self.csv_file=None
        super().__init__()

    def on_train_begin(self, logs=None):
        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename, "r") as f:
                    self.append_header = not bool(len(f.readline()))
            mode = "a"
        else:
            mode = "w"
        self.csv_file = open(self.filename, mode)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, str):
                return k
            elif (
                    isinstance(k, collections.Iterable)
                    and not is_zero_dim_ndarray
            ):
                return '"[%s]"' % (", ".join(map(str, k)))
            else:
                return k

        if self.keys is None:
            self.keys = sorted(logs.keys())

        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict(
                (k, logs[k]) if k in logs else (k, "NA") for k in self.keys
            )

        if not self.writer:

            class CustomDialect(csv.excel):
                delimiter = self.sep

            fieldnames = ["epoch"] + self.keys

            self.writer = csv.DictWriter(
                self.csv_file, fieldnames=fieldnames, dialect=CustomDialect
            )
            if self.append_header:
                self.writer.writeheader()

        row_dict = collections.OrderedDict({"epoch": epoch})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None


class ProgbarLogger(Callback):

    def __init__(self, max_epochs, steps, verbose=1):
        super().__init__()
        # Defaults to all Model's metrics except for loss.
        self.verbose = verbose
        self.seen = 0
        self.progbar = None
        self.target = steps
        self.epochs = max_epochs
        self._train_step, self._test_step, self._predict_step = None, None, None
        self._call_batch_hooks = True

        self._called_in_fit = False

    def on_train_begin(self, logs=None):
        # When this logger is called inside `fit`, validation is silent.
        self._called_in_fit = True

    def on_test_begin(self, logs=None):
        if not self._called_in_fit:
            self._reset_progbar()
            self._maybe_init_progbar()

    def on_predict_begin(self, logs=None):
        self._reset_progbar()
        self._maybe_init_progbar()

    def on_epoch_begin(self, epoch, logs=None):
        print_msg(f"Epoch {epoch + 1}/{self.epochs}")

    def on_train_batch_end(self, batch, logs=None):
        self._batch_update_progbar(batch, logs)

    def on_test_batch_end(self, batch, logs=None):
        if not self._called_in_fit:
            self._batch_update_progbar(batch, logs)

    def on_predict_batch_end(self, batch, logs=None):
        # Don't pass prediction results.
        self._batch_update_progbar(batch, None)

    def on_epoch_end(self, epoch, logs=None):
        self._finalize_progbar(logs, self._train_step)

    def on_test_end(self, logs=None):
        if not self._called_in_fit:
            self._finalize_progbar(logs, self._test_step)

    def on_predict_end(self, logs=None):
        self._finalize_progbar(logs, self._predict_step)

    def _reset_progbar(self):
        self.seen = 0
        self.progbar = None

    def _maybe_init_progbar(self):
        if self.progbar is None:
            self.progbar = Progbar(
                target=self.target,
                verbose=self.verbose,
                unit_name='step' if self.use_steps else 'sample')

    def _batch_update_progbar(self, batch, logs=None):
        """Updates the progbar."""
        logs = logs or {}
        self._maybe_init_progbar()
        self.seen = batch + 1  # One-indexed.
        self.progbar.update(self.seen, list(logs.items()), finalize=False)

    def _finalize_progbar(self, logs):
        logs = logs or {}
        if self.target is None:
            self.target = self.seen
        self.progbar.update(self.target, list(logs.items()), finalize=True)
