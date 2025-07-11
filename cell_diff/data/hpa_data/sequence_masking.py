# -*- coding: utf-8 -*-
from copy import deepcopy
import numpy as np
import random


def OAARDM_sequence_masking(
    protein_sequence: str,
    zero_mask_ratio: 0,
):
    seq_len = len(protein_sequence)
    
    if random.uniform(0, 1) < zero_mask_ratio:
        num_mask = 0
        zm_label = True
    else:
        num_mask = np.random.randint(1, seq_len + 1)
        zm_label = False
        
    mask_idc = np.random.choice(seq_len, num_mask, replace=False)

    protein_sequence_list = list(protein_sequence)
    protein_sequence_masked_list = deepcopy(protein_sequence_list)

    for idx in mask_idc:
        protein_sequence_masked_list[idx] = '<mask>'
    
    mask = np.full(seq_len, False)
    mask[mask_idc] = True

    protein_sequence_masked = ''.join(protein_sequence_masked_list)

    return protein_sequence_masked, mask, zm_label

if __name__ == "__main__":
    protein_sequence_masked, mask = OAARDM_sequence_masking("AAAALMLMLMLMAAA")
    print(protein_sequence_masked)
    print(mask)