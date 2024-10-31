# -*- coding: utf-8 -*-
from dataclasses import dataclass


@dataclass
class CELLDiffConfig:
    model_type: str = 'simple_diffusion'
    transformer_type: str = 'vit'

    # dataset
    data_path: str = ""
    num_val: int = 1000
    num_test: int = 1000
    img_crop_method: str = 'center'
    img_resize: int = 1024
    img_crop_size: int = 256
    seq_zero_mask_ratio: float = 0
    cell_image: str = 'nucl' # 'nucl', 'nucl,er', 'nucl,mt', 'nucl,er,mt'
    test_cell_image: str = 'nucl'
    split_key: str = 'train'

    # Diffusion params
    num_timesteps: int = 1000
    ddpm_beta_start: float = 0.0001
    ddpm_beta_end: float = 0.02
    ddpm_schedule: str = 'sigmoid'
    diffusion_pred_type: str = 'noise'
    timestep_respacing: str = ""

    # loss params
    sequence_loss_coeff: float = 1.0
    image_loss_coeff: float = 1.0

    # model params
    esm_embedding: str = 'esm2'
    embed_dim: int = 320
    esm_fixed_embedding: bool = True
    max_protein_sequence_len: int = 2048
    dit_patch_size: int = 16
    img_in_chans: int = 1
    depth: int = 16
    num_heads: int = 8
    mlp_ratio: float = 4.0
    num_residues: int = 32
    use_rna_expression: bool = False
    res_att_num_patches: int = 8
    attn_drop: float = 0.1
    cross_attn: bool = True
    cell_image_ratio: float = 1.0
    num_res_block: str = '2,2,2'
    dims: str = '32,64,128,256'
    
    # training
    loadcheck_path: str = ''
    ft: bool = False
    infer: bool = False
    
    # evaluation
    seq2img_n_samples: int = 1