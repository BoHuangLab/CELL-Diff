# -*- coding: utf-8 -*-
from dataclasses import dataclass
from transformers import PretrainedConfig

@dataclass
class VAEConfig(PretrainedConfig):
    model_type: str = 'vae'

    # Dataset parameters
    data_path: str = ""
    split_key: str = 'train'

    img_crop_method: str = 'center'
    img_resize: int = 1024
    img_crop_size: int = 256    
    seq_zero_mask_ratio: float = 0.0

    # Loss parameters
    recon_loss_coeff: float = 1.0
    kl_loss_coeff: float = 1.0

    # Model parameters
    in_channels: int = 1
    out_channels: int = 1
    num_down_blocks: int = 3
    latent_channels: int = 4
    vae_block_out_channels: str = '128,256,512'
    max_protein_sequence_len: int = 2048
    
    # Training parameters
    vae_loadcheck_path: str = ""
    ft: bool = False
    infer: bool = False
    ifresume: bool = False

    output_dir: str = ""
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    gradient_accumulation_steps: int = 1
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2

    num_train_epochs: int = 10
    fp16: bool = False
    bf16: bool = False
    logging_dir: str = ""
    logging_steps: int = 10
    max_steps: int = -1
    warmup_steps: int = 1000
    save_steps: int = 1000

    dataloader_num_workers: int = 8
    seed: int = 6
    
    # evaluation
    seq2img_n_samples: int = 1


    def __init__(self, **kwargs):
        # Use `super().__init__` to handle arguments from PretrainedConfig
        super().__init__(**kwargs)

        self.model_type = kwargs.get("model_type", self.model_type)

        # Initialize all custom attributes from `kwargs` or defaults
        self.data_path = kwargs.get("data_path", self.data_path)
        self.split_key = kwargs.get("split_key", self.split_key)

        self.img_crop_method = kwargs.get("img_crop_method", self.img_crop_method)
        self.img_resize = kwargs.get("img_resize", self.img_resize)
        self.img_crop_size = kwargs.get("img_crop_size", self.img_crop_size)
        self.seq_zero_mask_ratio = kwargs.get("seq_zero_mask_ratio", self.seq_zero_mask_ratio)

        self.recon_loss_coeff = kwargs.get("recon_loss_coeff", self.recon_loss_coeff)
        self.kl_loss_coeff = kwargs.get("kl_loss_coeff", self.kl_loss_coeff)
    
        self.in_channels = kwargs.get("in_channels", self.in_channels)
        self.out_channels = kwargs.get("out_channels", self.out_channels)
        self.num_down_blocks = kwargs.get("num_down_blocks", self.num_down_blocks)
        self.latent_channels = kwargs.get("latent_channels", self.latent_channels)
        
        self.vae_block_out_channels = kwargs.get("vae_block_out_channels", self.vae_block_out_channels)
        self.vae_block_out_channels = [int(a) for a in self.vae_block_out_channels.split(',')]

        self.max_protein_sequence_len = kwargs.get("max_protein_sequence_len", self.max_protein_sequence_len)

        self.vae_loadcheck_path = kwargs.get("vae_loadcheck_path", self.vae_loadcheck_path)
        self.ft = kwargs.get("ft", self.ft)
        self.infer = kwargs.get("infer", self.infer)
        self.ifresume = kwargs.get("ifresume", self.ifresume)

        self.output_dir = kwargs.get("output_dir", self.output_dir)
        self.learning_rate = kwargs.get("learning_rate", self.learning_rate)
        self.weight_decay = kwargs.get("weight_decay", self.weight_decay)
        self.gradient_accumulation_steps = kwargs.get("gradient_accumulation_steps", self.gradient_accumulation_steps)
        self.per_device_train_batch_size = kwargs.get("per_device_train_batch_size", self.per_device_train_batch_size)
        self.per_device_eval_batch_size = kwargs.get("per_device_eval_batch_size", self.per_device_eval_batch_size)

        self.num_train_epochs = kwargs.get("num_train_epochs", self.num_train_epochs)
        self.fp16 = kwargs.get("fp16", self.fp16)
        self.bf16 = kwargs.get("bf16", self.bf16)
        self.logging_dir = kwargs.get("logging_dir", self.logging_dir)
        self.logging_steps = kwargs.get("logging_steps", self.logging_steps)
        self.max_steps = kwargs.get("max_steps", self.max_steps)
        self.warmup_steps = kwargs.get("warmup_steps", self.warmup_steps)
        self.save_steps = kwargs.get("save_steps", self.save_steps)

        self.dataloader_num_workers = kwargs.get("dataloader_num_workers", self.dataloader_num_workers)
        self.seed = kwargs.get("seed", self.seed)
