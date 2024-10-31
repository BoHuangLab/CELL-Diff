# -*- coding: utf-8 -*-
from typing import List, Tuple

from cell_diff.logging.loggers import logger
from cell_diff.utils.optim.adam import AdamW as Adam

def split_param_and_layer_name(name_list: List[str]) -> Tuple[List[str], List[int]]:
    param_list = []
    layer_name_list = []
    for name in name_list:
        if isinstance(name, str):
            param_list.append(name)
        elif isinstance(name, int):
            layer_name_list.append(name)
        else:
            raise ValueError(f"Invalid name type: {type(name)}")

    return param_list, layer_name_list

def process_param(
    net,
    freeze_list: List = [],
    unfreeze_list: List = [],
    lr: float = 1e-5,
    mfm_lora: bool = False,
    **kwargs,
):
    param_groups = [{}]
    param_groups[0]["lr"] = lr
    param_groups[0]["params"] = []

    if len(unfreeze_list) > 0:
        unfreeze_list, unfreeze_layer_name_list = split_param_and_layer_name(
            unfreeze_list
        )
        for name, param in net.named_parameters():
            nl = int(name.split(".")[0])
            if nl in unfreeze_layer_name_list:
                param_groups[0]["params"].append(param)
                if name.find("dummy") == -1:
                    logger.success(f"unfreeze layer: {name}")
            else:
                for unfreeze_name in unfreeze_list:
                    if name.find(unfreeze_name) != -1:
                        param_groups[0]["params"].append(param)
                        if name.find("dummy") == -1:
                            logger.success(f"unfreeze layer: {name}")

    elif len(freeze_list) > 0:
        freeze_list, freeze_layer_name_list = split_param_and_layer_name(unfreeze_list)
        for name, param in net.named_parameters():
            nl = int(name.split(".")[0])
            if nl in freeze_layer_name_list:
                flag = True
                logger.success(f"freeze layer: {name}")
            else:
                for freeze_name in freeze_list:
                    flag = False
                    if name.find(freeze_name) != -1:
                        flag = True
                        logger.success(f"freeze {name}")
                        break
            if not flag:
                param_groups[0]["params"].append(param)

    else:
        for name, param in net.named_parameters():
            param_groups[0]["params"].append(param)

    for param_group in param_groups:
        if "lr" not in param_group:
            param_group["lr"] = kwargs["lr"]
        if "weight_decay" not in param_group:
            param_group["weight_decay"] = kwargs.get("weight_decay", 0.0)

    return param_groups


def myAdam(
    net,
    impl=Adam,
    freeze_list: List = [],
    unfreeze_list: List = [],
    mfm_lora=False,
    **kwargs,
):
    assert (
        len(freeze_list) == 0 or len(unfreeze_list) == 0
    ), "freeze_list and unfreeze_list cannot be set at the same time"

    new_param_groups = []
    param_groups = process_param(
        net,
        freeze_list=freeze_list,
        unfreeze_list=unfreeze_list,
        mfm_lora=mfm_lora,
        **kwargs,
    )
    for param_group in param_groups:
        new_param_groups.extend([param_group])
    return impl(new_param_groups, **kwargs), param_groups
