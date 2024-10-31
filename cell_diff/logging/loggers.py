# -*- coding: utf-8 -*-
import os
import sys
from dataclasses import dataclass, fields, is_dataclass
from typing import Dict, Union

import torch
from loguru import logger

from cell_diff.utils.dist_utils import is_master_node

import wandb  # isort:skip

handlers = {}


def get_logger():
    if not handlers:
        logger.remove()  # remove default handler
        handlers["console"] = logger.add(
            sys.stdout,
            format="[<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green>][<cyan>{level}</cyan>]: {message}",
            colorize=True,
            filter=console_log_filter,
            enqueue=True,
        )

    return logger


# Custom function to handle tensor attributes
def dataclass_to_dict(dataclass_obj: Union[dataclass, Dict]) -> Dict:
    if isinstance(dataclass_obj, dict):
        return dataclass_obj
    result = {}
    for field in fields(dataclass_obj):
        value = getattr(dataclass_obj, field.name)
        if isinstance(value, torch.Tensor) and not value.is_leaf:
            result[field.name] = value.clone().detach()
        else:
            result[field.name] = value
    return result


class MetricLogger(object):
    def log(self, metrics, prefix=""):
        if not is_master_node():
            return

        log_data = {}
        if type(metrics) is dict:
            log_data = metrics
        elif is_dataclass(metrics):
            log_data = dataclass_to_dict(metrics)

            if "extra_output" in log_data:
                extra_output = log_data["extra_output"]
                if extra_output is not None:
                    for k, v in extra_output.items():
                        log_data[k] = v
                del log_data["extra_output"]

        for k in log_data:
            if isinstance(log_data[k], torch.Tensor):
                log_data[k] = log_data[k].detach().item()

        logger.info(" | ".join([f"{k}={v:.4g}" for k, v in log_data.items()]))
        if wandb.run is not None:
            # Add prefix
            if prefix:
                log_data = {f"{prefix}/{k}": v for k, v in log_data.items()}
            wandb.log(log_data)


def console_log_filter(record):
    # For message with level INFO, we only log it on master node
    # For others, we log it on all nodes
    if record["level"].name != "INFO":
        return True

    return is_master_node()
