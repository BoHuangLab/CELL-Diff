# -*- coding: utf-8 -*-
import os

import torch


def is_master_node():
    if "RANK" not in os.environ or int(os.environ["RANK"]) == 0:
        return True
    else:
        return False


def support_flash_attntion():
    # check the GPU is A100 or H100
    device_name = torch.cuda.get_device_name(0)
    for name in ["A100", "H100"]:
        if name in device_name:
            return True
    return False
