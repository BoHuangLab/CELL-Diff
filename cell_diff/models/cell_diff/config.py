# -*- coding: utf-8 -*-
from dataclasses import dataclass
from transformers import PretrainedConfig

@dataclass
class CELLDiffConfig(PretrainedConfig):
    model_type: str = 'latent_diffusion'
    transformer_type: str = 'vit'

    # Dataset parameters
    data_path: str = ""
    split_key: str = 'train'

    img_crop_method: str = 'center'
    img_resize: int = 1024
    img_crop_size: int = 256
    seq_zero_mask_ratio: float = 0
    
    cell_image: str = 'nucl' # 'nucl', 'nucl,er', 'nucl,mt', 'nucl,er,mt'
    test_cell_image: str = 'nucl' # 'nucl', 'nucl,er', 'nucl,mt', 'nucl,er,mt'

    # Diffusion parameters
    num_timesteps: int = 1000
    ddpm_beta_start: float = 0.0001
    ddpm_beta_end: float = 0.02
    ddpm_schedule: str = 'sigmoid'
    diffusion_pred_type: str = 'noise'
    timestep_respacing: str = ""

    # Model parameters
    ## VAE
    in_channels: int = 1
    out_channels: int = 1
    num_down_blocks: int = 3
    latent_channels: int = 4
    vae_block_out_channels: str = '128,256,512'

    ## CELL-Diff
    block_out_channels: str = '320,640,1280,1280'
    layers_per_block: int = 2
    mid_num_attention_heads: int = 16
    sample_size: int = 64
    esm_embedding: str = 'esm2'
    esm_fixed_embedding: bool = True
    hidden_size: int = 320
    max_protein_sequence_len: int = 2048
    num_hidden_layers: int = 16
    num_attention_heads: int = 8
    mlp_ratio: float = 4.0
    vocab_size: int = 32
    attn_drop: float = 0.0
    output_rescale: bool = False

    dit_patch_size: int = 1
    cell_image_ratio: float = 1.0
    cross_attn: bool = True

    # Loss parameters
    sequence_loss_coeff: float = 1.0
    image_loss_coeff: float = 1.0

    # Training parameters
    vae_loadcheck_path: str = '.'
    loadcheck_path: str = '.'
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
    
    # Evaluation
    image_path: str = None
    test_sequence: str = None
    num_aas: int = 20

    def __init__(self, **kwargs):
        # Use `super().__init__` to handle arguments from PretrainedConfig
        super().__init__(**kwargs)

        self.model_type = kwargs.get("model_type", self.model_type)
        self.transformer_type = kwargs.get("transformer_type", self.transformer_type)

        # Initialize all custom attributes from `kwargs` or defaults
        self.data_path = kwargs.get("data_path", self.data_path)
        self.split_key = kwargs.get("split_key", self.split_key)

        self.img_crop_method = kwargs.get("img_crop_method", self.img_crop_method)
        self.img_resize = kwargs.get("img_resize", self.img_resize)
        self.img_crop_size = kwargs.get("img_crop_size", self.img_crop_size)
        self.seq_zero_mask_ratio = kwargs.get("seq_zero_mask_ratio", self.seq_zero_mask_ratio)
        
        self.cell_image = kwargs.get("cell_image", self.cell_image)
        self.test_cell_image = kwargs.get("test_cell_image", self.test_cell_image)

        self.num_timesteps = kwargs.get("num_timesteps", self.num_timesteps)
        self.ddpm_beta_start = kwargs.get("ddpm_beta_start", self.ddpm_beta_start)
        self.ddpm_beta_end = kwargs.get("ddpm_beta_end", self.ddpm_beta_end)
        self.ddpm_schedule = kwargs.get("ddpm_schedule", self.ddpm_schedule)
        self.diffusion_pred_type = kwargs.get("diffusion_pred_type", self.diffusion_pred_type)
        self.timestep_respacing = kwargs.get("timestep_respacing", self.timestep_respacing)

        self.in_channels = kwargs.get("in_channels", self.in_channels)
        self.out_channels = kwargs.get("out_channels", self.out_channels)
        self.num_down_blocks = kwargs.get("num_down_blocks", self.num_down_blocks)
        self.latent_channels = kwargs.get("latent_channels", self.latent_channels)
        self.vae_block_out_channels = kwargs.get("vae_block_out_channels", self.vae_block_out_channels)
        if not isinstance(self.vae_block_out_channels, list):
            self.vae_block_out_channels = [int(a) for a in self.vae_block_out_channels.split(',')]

        self.block_out_channels = kwargs.get("block_out_channels", self.block_out_channels)
        if not isinstance(self.block_out_channels, list):
            self.block_out_channels = [int(a) for a in self.block_out_channels.split(',')]
        
        self.layers_per_block = kwargs.get("layers_per_block", self.layers_per_block)
        self.mid_num_attention_heads = kwargs.get("mid_num_attention_heads", self.mid_num_attention_heads)
        self.sample_size = kwargs.get("sample_size", self.sample_size)
        self.esm_embedding = kwargs.get("esm_embedding", self.esm_embedding)
        self.esm_fixed_embedding = kwargs.get("esm_fixed_embedding", self.esm_fixed_embedding)
        self.hidden_size = kwargs.get("hidden_size", self.hidden_size)
        self.max_protein_sequence_len = kwargs.get("max_protein_sequence_len", self.max_protein_sequence_len)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", self.num_hidden_layers)
        self.num_attention_heads = kwargs.get("num_attention_heads", self.num_attention_heads)
        self.mlp_ratio = kwargs.get("mlp_ratio", self.mlp_ratio)
        self.vocab_size = kwargs.get("vocab_size", self.vocab_size)
        self.attn_drop = kwargs.get("attn_drop", self.attn_drop)
        self.output_rescale = kwargs.get("output_rescale", self.output_rescale)

        self.dit_patch_size = kwargs.get("dit_patch_size", self.dit_patch_size)
        self.cell_image_ratio = kwargs.get("cell_image_ratio", self.cell_image_ratio)
        self.cross_attn = kwargs.get("cross_attn", self.cross_attn)

        self.sequence_loss_coeff = kwargs.get("sequence_loss_coeff", self.sequence_loss_coeff)
        self.image_loss_coeff = kwargs.get("image_loss_coeff", self.image_loss_coeff)
        
        self.vae_loadcheck_path = kwargs.get("vae_loadcheck_path", self.vae_loadcheck_path)
        self.loadcheck_path = kwargs.get("loadcheck_path", self.loadcheck_path)
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

        self.image_path = kwargs.get("image_path", self.image_path)
        self.test_sequence = kwargs.get("test_sequence", self.test_sequence)
        self.num_aas = kwargs.get("num_aas", self.num_aas)
