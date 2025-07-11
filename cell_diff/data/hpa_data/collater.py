# -*- coding: utf-8 -*-
from typing import List, Union
import torch
from .vocabulary import Alphabet


# allow pad_num to be int or float
def pad_1d_unsqueeze(
    x: torch.Tensor, padlen: int, start: int, pad_num: Union[int, float]
):
    # (N) -> (1, padlen)
    xlen = x.size(0)
    assert (
        start + xlen <= padlen
    ), f"padlen {padlen} is too small for xlen {xlen} and start point {start}"
    new_x = x.new_full([padlen], pad_num, dtype=x.dtype)
    new_x[start : start + xlen] = x
    x = new_x
    return x.unsqueeze(0)


def collate_fn(samples: List[dict], 
               vocab: Alphabet, 
               max_protein_sequence_len: int, 
               min_protein_sequence_len: int = 0, 
               vae = None):
    """
    Overload BaseWrapperDataset.collater
    May be future changes need config

    By default, the collater pads and batch all torch.Tensors (np.array will be converted) in the sample dicts
    """
    # max_tokens = Nres+2 (<cls> and <eos>)
    
    samples = [
        s
        for s in samples
        if s["protein_seq"].size(0) <= (max_protein_sequence_len + 2) and s["protein_seq"].size(0) > (min_protein_sequence_len + 2)
    ]

    max_tokens = max(len(s["protein_seq"]) for s in samples)

    batch = dict()

    # (Nres+2,) -> (B, Nres+2)
    batch["protein_seq"] = torch.cat(
        [
            pad_1d_unsqueeze(
                s["protein_seq"], max_tokens, 0, vocab.padding_idx
            )
            for s in samples
        ]
    ).unsqueeze(-1)

    batch["protein_seq_masked"] = torch.cat(
        [
            pad_1d_unsqueeze(
                s["protein_seq_masked"], max_tokens, 0, vocab.padding_idx
            )
            for s in samples
        ]
    ).unsqueeze(-1)

    batch["protein_seq_mask"] = torch.cat(
        [
            pad_1d_unsqueeze(
                s["protein_seq_mask"], max_tokens, 0, 0)
            for s in samples
        ]
    ).unsqueeze(-1)

    batch["protein_img"] = torch.cat(
        [s["protein_img"].unsqueeze(0) for s in samples]
    )

    batch["nucleus_img"] = torch.cat(
        [s["nucleus_img"].unsqueeze(0) for s in samples]
    )

    batch["microtubules_img"] = torch.cat(
        [s["microtubules_img"].unsqueeze(0) for s in samples]
    )

    batch["ER_img"] = torch.cat(
        [s["ER_img"].unsqueeze(0) for s in samples]
    )

    batch["rna_expression"] = torch.cat(
        [s["rna_expression"].unsqueeze(0) for s in samples]
    )

    batch["zm_label"] = torch.cat(
        [s["zm_label"].unsqueeze(0) for s in samples]
    )

    return {'batched_data': batch, 'vae': vae}