# -*- coding: utf-8 -*-
import os
from typing import List

import lmdb
import numpy as np
import torch

from torch.utils.data import Dataset

from .collater import collate_fn
from cell_diff.data.hpa_data.vocabulary import Alphabet, convert_string_sequence_to_int_index
from cell_diff.data.hpa_data.sequence_masking import OAARDM_sequence_masking
from cell_diff.data.hpa_data.img_utils import RandomRotation

from torchvision import transforms

import json
import pickle

class OpenCellLMDBDataset(Dataset):
    def __init__(self, args, split_key, vae) -> None:
        super().__init__()
        self.args = args

        self.split_key = split_key
        self.data_path = self.args.data_path
        self.vocab = Alphabet()

        with open(os.path.join(self.data_path, f'{self.split_key}_keys.json')) as f:
            self.data_files = json.load(f)

        self.img_crop_method = self.args.img_crop_method
        self.img_resize = self.args.img_resize
        self.img_crop_size = self.args.img_crop_size

        self.vae = vae
        self.env = None

    def init_env(self):
        self.env = lmdb.open(self.data_path, readonly=True, max_readers=256, lock=False).begin()

    def __getitem__(self, index: int) -> dict:
        data_file = self.data_files[index]
        
        item = {}

        if self.env is None:
            self.init_env()
        
        data = self.env.get(data_file.encode())
        data = pickle.loads(data)

        item['protein_img'], item['nucleus_img'] = self.get_img(data)
        
        protein_seq = data["protein_seq"]
        item['cell_line'] = data["cell_line"]
        item['rna_expression'] = torch.FloatTensor([data["rna_expression"]])

        protein_seq_masked, protein_seq_mask, zm_label = OAARDM_sequence_masking(protein_seq, self.args.seq_zero_mask_ratio)

        """
        - convert string sequence to int index
        """
        protein_seq_token = convert_string_sequence_to_int_index(self.vocab, protein_seq)
        protein_seq_masked_token = convert_string_sequence_to_int_index(self.vocab, protein_seq_masked)

        if self.vocab.prepend_bos:
            protein_seq_mask = np.insert(protein_seq_mask, 0, False)
        if self.vocab.append_eos:
            protein_seq_mask = np.append(protein_seq_mask, False)
        
        item['zm_label'] = torch.Tensor([zm_label]).bool()
        item['protein_seq'] = torch.LongTensor(protein_seq_token)
        item['protein_seq_masked'] = torch.LongTensor(protein_seq_masked_token)
        item['protein_seq_mask'] = torch.from_numpy(protein_seq_mask).long()
        item['prot_id'] = data['prot_id']

        return item

    def get_img(self, item):        
        img = item['img'].astype(np.float32)
        img = np.clip(img, 500, 65535)
        img = (img - 500) / (65535 - 500)
        
        img = torch.from_numpy(img)
        
        nucleus_img = img[0]
        protein_img = img[1]
        
        # Rescale nucleus image
        nucleus_img = (nucleus_img * 3).clamp(0, 1)
        # Rescale protein image
        protein_img = (protein_img * 3).clamp(0, 1)
        
        # To maintain the resolution.
        # First crop then resize.
        
        t_forms = []

        if self.img_crop_method == 'random':
            t_forms.append(transforms.RandomCrop(self.img_crop_size))
            t_forms.append(transforms.RandomHorizontalFlip(p=0.5))
            t_forms.append(RandomRotation([0, 90, 180, 270]))

        elif self.img_crop_method == 'center':
            t_forms.append(transforms.CenterCrop(self.img_crop_size))

        t_forms.append(transforms.Resize(self.img_resize, antialias=None))

        t_forms.append(transforms.Normalize(mean=[0.5], std=[0.5]))
        t_forms = transforms.Compose(t_forms)
        
        img = torch.stack([nucleus_img, protein_img], dim=0)
        
        nucleus_img, protein_img = t_forms(img)
                
        return protein_img.unsqueeze(0), nucleus_img.unsqueeze(0)

    def __len__(self) -> int:
        return len(self.data_files)

    def collate(self, samples: List[dict]) -> dict:
        return collate_fn(samples, self.vocab, self.args.max_protein_sequence_len, 0, self.vae)


class OpenCellSecLMDBDataset(OpenCellLMDBDataset):
    def get_img(self, item):        
        img = item['img'].astype(np.float32)
        img = np.clip(img, 500, 65535)
        img = (img - 500) / (65535 - 500)
        
        img = torch.from_numpy(img)
        
        nucleus_img = img[0]
        protein_img = img[1]
        
        # Rescale nucleus image
        nucleus_img = (nucleus_img * 3).clamp(0, 1)
        # Rescale protein image
        protein_img = (protein_img * 3).clamp(0, 1)
        
        # To maintain the resolution.
        # First crop then resize.
        
        t_forms = []

        if self.img_crop_method == 'random':
            t_forms.append(transforms.RandomCrop(self.img_crop_size))
            t_forms.append(transforms.RandomHorizontalFlip(p=0.5))
            t_forms.append(RandomRotation([0, 90, 180, 270]))

        elif self.img_crop_method == 'center':
            t_forms.append(transforms.CenterCrop(self.img_crop_size))

        t_forms.append(transforms.Resize(self.img_resize, antialias=None))

        t_forms.append(transforms.Normalize(mean=[0.5], std=[0.5]))
        t_forms = transforms.Compose(t_forms)
        
        img = torch.stack([nucleus_img, protein_img], dim=0)
        
        nucleus_img, protein_img = t_forms(img)
                
        return protein_img.unsqueeze(0), nucleus_img.unsqueeze(0)

