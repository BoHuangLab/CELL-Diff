import os
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Union
from cell_diff.logging import logger

from diffusers.models.embeddings import PatchEmbed
from diffusers.models.embeddings import TimestepEmbedding, GaussianFourierProjection, Timesteps
from diffusers.models.unets.unet_2d_blocks import get_down_block, get_up_block
from diffusers.models.activations import get_activation

from .modules.protein_sequence_embedding import ESMEmbed
from .modules.diffusion import create_diffusion
from .modules.transformer import TransformerBlock, unpatchify, FinalLayer
from .modules.positional_embedding import get_1d_sincos_pos_embed

from transformers import PreTrainedModel
from .config import CELLDiffConfig

from cell_diff.pipeline.utils import CELLDiffOutput
from cell_diff.models.vae.vae_model import VAEModel
import json


class CELLDiffModel(PreTrainedModel):
    config_class = CELLDiffConfig

    def __init__(self, config, loss_fn=None):
        super().__init__(config)
        self.config = config
        
        self.loss = loss_fn(config)

        self.cell_img_placeholder = nn.Parameter(
            torch.randn(size=(config.sample_size, config.sample_size), requires_grad=True)
        )

        self.net = CELLDiff(config)

        predict_xstart = True if self.config.diffusion_pred_type == 'xstart' else False
        self.diffusion = create_diffusion(
            timestep_respacing=config.timestep_respacing, 
            noise_schedule=config.ddpm_schedule, 
            learn_sigma=False, 
            image_d=config.sample_size, 
            predict_xstart=predict_xstart, 
            diffusion_steps=config.num_timesteps, 
        )
        self.load_pretrained_weights(config, checkpoint_path=config.loadcheck_path)

    def load_pretrained_weights(self, config, checkpoint_path):
        """
        Load pretrained weights from a given state_dict.
        """
        if config.ft or config.infer:
            if config.ft:
                logger.info(f"Finetune from checkpoint: {checkpoint_path}")
            else:
                logger.info(f"Infer from checkpoint: {checkpoint_path}")
                
            if os.path.isdir(checkpoint_path):
                with open(os.path.join(checkpoint_path, "pytorch_model.bin.index.json"), "r") as f:
                    index_data = json.load(f)

                weight_map = index_data["weight_map"]

                shard_files = set(weight_map.values())
                shard_weights = {}
                for shard_file in shard_files:
                    shard_weights[shard_file] = torch.load(os.path.join(checkpoint_path, shard_file), map_location="cpu")

                checkpoints_state = {}
                for param_name, shard_file in weight_map.items():
                    checkpoints_state[param_name] = shard_weights[shard_file][param_name]
            elif os.path.splitext(checkpoint_path)[1] == '.safetensors':
                from safetensors.torch import load_file
                checkpoints_state = load_file(checkpoint_path)
            else:
                checkpoints_state = torch.load(checkpoint_path, map_location="cpu")

            if "model" in checkpoints_state:
                checkpoints_state = checkpoints_state["model"]
            elif "module" in checkpoints_state:
                checkpoints_state = checkpoints_state["module"]

            model_state_dict = self.state_dict()
            filtered_state_dict = {k: v for k, v in checkpoints_state.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}

            IncompatibleKeys = self.load_state_dict(filtered_state_dict, strict=False)
            IncompatibleKeys = IncompatibleKeys._asdict()

            missing_keys = []
            for keys in IncompatibleKeys["missing_keys"]:
                if keys.find("dummy") == -1:
                    missing_keys.append(keys)

            unexpected_keys = []
            for keys in IncompatibleKeys["unexpected_keys"]:
                if keys.find("dummy") == -1:
                    unexpected_keys.append(keys)

            if len(missing_keys) > 0:
                logger.info(
                    "Missing keys in {}: {}".format(
                        checkpoint_path,
                        missing_keys,
                    )
                )

            if len(unexpected_keys) > 0:
                logger.info(
                    "Unexpected keys {}: {}".format(
                        checkpoint_path,
                        unexpected_keys,
                    )
                )

    def forward(self, batched_data, vae: VAEModel, **kwargs):

        with torch.no_grad():
            protein_img = batched_data['protein_img']

            protein_img_latent = vae.encode(protein_img).sample()
            nucleus_img_latent = vae.encode(batched_data['nucleus_img']).sample()

            if self.config.cell_image == 'nucl':
                cell_img_latent = nucleus_img_latent

            elif self.config.cell_image == 'nucl,er':
                if 'ER_img' in batched_data:
                    ER_img = batched_data['ER_img']
                    ER_img_latent = vae.encode(ER_img).sample()
                    mask = (torch.rand(ER_img_latent.shape[0]) <= self.config.cell_image_ratio).bool()
                    ER_img_latent[~mask] = self.cell_img_placeholder
                else:
                    ER_img_latent = self.cell_img_placeholder.expand(
                        protein_img_latent.shape[0], 
                        protein_img_latent.shape[1], 
                        *self.cell_img_placeholder.shape
                    )
                cell_img_latent = torch.cat([nucleus_img_latent, ER_img_latent], dim=1)

            elif self.config.cell_image == 'nucl,mt':
                if 'microtubules_img' in batched_data:
                    microtubules_img = batched_data['microtubules_img']
                    microtubules_img_latent = vae.encode(microtubules_img).sample()
                    mask = (torch.rand(microtubules_img_latent.shape[0]) <= self.config.cell_image_ratio).bool()
                    microtubules_img_latent[~mask] = self.cell_img_placeholder
                else:
                    microtubules_img_latent = self.cell_img_placeholder.expand(
                        protein_img_latent.shape[0], 
                        protein_img_latent.shape[1], 
                        *self.cell_img_placeholder.shape
                    )
                cell_img_latent = torch.cat([nucleus_img_latent, microtubules_img_latent], dim=1)

            elif self.config.cell_image == 'nucl,er,mt':
                if 'ER_img' in batched_data:
                    ER_img = batched_data['ER_img']
                    ER_img_latent = vae.encode(ER_img).sample()
                    mask = (torch.rand(ER_img_latent.shape[0]) <= self.config.cell_image_ratio).bool()
                    ER_img_latent[~mask] = self.cell_img_placeholder
                else:
                    ER_img_latent = self.cell_img_placeholder.expand(
                        protein_img_latent.shape[0], 
                        protein_img_latent.shape[1], 
                        *self.cell_img_placeholder.shape
                    )

                if 'microtubules_img' in batched_data:
                    microtubules_img = batched_data['microtubules_img']
                    microtubules_img_latent = vae.encode(microtubules_img).sample()
                    mask = (torch.rand(microtubules_img_latent.shape[0]) <= self.config.cell_image_ratio).bool()
                    microtubules_img_latent[~mask] = self.cell_img_placeholder
                else:
                    microtubules_img_latent = self.cell_img_placeholder.expand(
                        protein_img_latent.shape[0], 
                        protein_img_latent.shape[1], 
                        *self.cell_img_placeholder.shape
                    )

                cell_img_latent = torch.cat([nucleus_img_latent, ER_img_latent, microtubules_img_latent], dim=1)
            else:
                raise ValueError(f"Cell image type: {self.config.cell_image} is not supported")

        # add noise to protein_img
        time = torch.randint(
            0, self.diffusion.num_timesteps, (protein_img_latent.shape[0],), device=protein_img_latent.device, 
        )

        zm_label = batched_data['zm_label'].squeeze(-1).bool()
        time[~zm_label] = 0

        noise = torch.randn_like(protein_img_latent)
        protein_img_latent_noisy = self.diffusion.q_sample(protein_img_latent, time, noise)
        sqrt_one_minus_alphas_cumprod_t = self.diffusion.sqrt_one_minus_alphas_cumprod_t(protein_img_latent, time)
        model_time = self.diffusion._scale_timesteps(time)

        protein_img_latent_output, seq_output = self.net(
            protein_img_latent_noisy, cell_img_latent, 
            batched_data['protein_seq_masked'], model_time, 
            sqrt_one_minus_alphas_cumprod_t
        )

        diff_loss_dict = self.diffusion.training_losses(
            protein_img_latent_output.to(torch.float32), protein_img_latent_noisy.to(torch.float32), 
            protein_img_latent.to(torch.float32), time, noise, 
        )
        
        model_output = (protein_img_latent_output, seq_output, diff_loss_dict['loss'])
        loss, log_loss = self.loss(batched_data, model_output)

        return CELLDiffOutput(loss=loss, log_output=log_loss)

    def sequence_to_image(self, protein_seq, cell_img_latent, progress=True, sampling_strategy="ddpm"):
        protein_img_latent = torch.randn(
            cell_img_latent.shape[0], self.config.latent_channels, cell_img_latent.shape[2], cell_img_latent.shape[3]
        ).to(cell_img_latent.device)
        indices = list(range(self.diffusion.num_timesteps))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)
        
        for i in indices:
            time = torch.tensor([i] * cell_img_latent.shape[0], device=cell_img_latent.device)
            with torch.no_grad():
                model_time = self.diffusion._scale_timesteps(time)
                sqrt_one_minus_alphas_cumprod_t = self.diffusion.sqrt_one_minus_alphas_cumprod_t(protein_img_latent, time)
                protein_img_latent_output = self.net(
                    protein_img_latent, cell_img_latent, protein_seq, model_time, sqrt_one_minus_alphas_cumprod_t
                )[0]

                if sampling_strategy == "ddpm":
                    out = self.diffusion.p_sample(
                        protein_img_latent_output, protein_img_latent, time, clip_denoised=False, 
                    )
                    protein_img_latent = out["sample"]
                elif sampling_strategy == "ddim":
                    out = self.diffusion.ddim_sample(
                        protein_img_latent_output, protein_img_latent, time, clip_denoised=False, 
                    )
                    protein_img_latent = out["sample"]

        return protein_img_latent

    def image_to_sequence(self, protein_seq, protein_seq_mask, protein_img_latent, cell_img_latent, progress=True, sampling_strategy='oaardm', order='l2r', temperature=1.0):
        if sampling_strategy == "oaardm":
            return self.oaardm_sample(protein_seq, protein_seq_mask, protein_img_latent, cell_img_latent, progress, order, temperature)

    def oaardm_sample(self, protein_seq, protein_seq_mask, protein_img_latent, cell_img_latent, progress=True, order='random', temperature=1.0):
        loc = torch.where(protein_seq_mask.bool().squeeze())[0].cpu().numpy()

        if order == 'l2r': 
            loc = np.sort(loc)
        elif order == 'random': 
            np.random.shuffle(loc)
        
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            loc = tqdm(loc)

        with torch.no_grad():
            for i in loc:
                time = torch.tensor([0] * protein_seq.shape[0], device=protein_seq.device)
                model_time = self.diffusion._scale_timesteps(time)
                sqrt_one_minus_alphas_cumprod_t = self.diffusion.sqrt_one_minus_alphas_cumprod_t(protein_img_latent, time)

                seq_output = self.net(
                    protein_img_latent, cell_img_latent, protein_seq, model_time, sqrt_one_minus_alphas_cumprod_t
                )[1]
                
                p = seq_output[:, i, 4:4+20] # sample at location i (random), dont let it predict non-standard AA
                p = torch.nn.functional.softmax(p / temperature, dim=1) # softmax over categorical probs
                p_sample = torch.multinomial(p, num_samples=1)
                protein_seq[:, i] = p_sample.squeeze() + 4

        return protein_seq


class CELLDiff(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

        num_attention_heads = args.num_attention_heads
        attention_head_dim = args.num_attention_heads
        cross_attention_dim = args.hidden_size
        layers_per_block = args.layers_per_block
        transformer_layers_per_block = 1
        block_out_channels = args.block_out_channels

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        protein_sequence_pos_embed = get_1d_sincos_pos_embed(
            args.hidden_size, self.args.max_protein_sequence_len + 2, 
        )
        # Use fixed sin-cos embedding:
        self.protein_sequence_pos_embed = nn.Parameter(
            torch.from_numpy(protein_sequence_pos_embed).float().unsqueeze(0), 
            requires_grad=False, 
        )
        self.seq_proj_in = nn.Linear(args.hidden_size, args.hidden_size, bias=True)

        # Use ConvNet to process image
        self.img_conv_in = nn.Conv2d(
            args.latent_channels*(1 + len(args.cell_image.split(','))), 
            block_out_channels[0], 3, 1, 1,
        )

        # time
        time_embed_dim, timestep_input_dim = self._set_time_proj(
            time_embedding_type="positional", 
            block_out_channels=block_out_channels, 
            flip_sin_to_cos=True, 
            freq_shift=0, 
            time_embedding_dim=None,
        )

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim, time_embed_dim, act_fn='silu', 
        )

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        if args.cross_attn:
            down_block_types = (
                "CrossAttnDownBlock2D", 
                "CrossAttnDownBlock2D", 
                "CrossAttnDownBlock2D", 
                "DownBlock2D", 
            )
        else:
            down_block_types = (
                "DownBlock2D", 
                "DownBlock2D", 
                "DownBlock2D", 
                "DownBlock2D", 
            )

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)

        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block[i],
                transformer_layers_per_block=transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=1e-5,
                resnet_act_fn="silu",
                resnet_groups=32,
                cross_attention_dim=cross_attention_dim[i],
                num_attention_heads=num_attention_heads[i],
                downsample_padding=1,
                dual_cross_attention=False,
                use_linear_projection=False,
                only_cross_attention=False,
                upcast_attention=False,
                resnet_time_scale_shift="default",
                attention_type="default",
                resnet_skip_time_act=False,
                resnet_out_scale_factor=1.0, 
                cross_attention_norm=None,
                attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,
                dropout=args.attn_drop,
            )
            self.down_blocks.append(down_block)
        
        # mid
        self.img_embedding = PatchEmbed(
            height=args.sample_size // (2 ** (len(block_out_channels) - 1)), 
            width=args.sample_size // (2 ** (len(block_out_channels) - 1)), 
            patch_size=args.dit_patch_size, 
            in_channels=block_out_channels[-1], 
            embed_dim=args.hidden_size, 
        )
        
        self.token_type_embeddings = nn.Embedding(2, args.hidden_size)

        self.unified_encoder = nn.ModuleList([
            TransformerBlock(
                args.hidden_size, 
                args.mid_num_attention_heads, 
                mlp_ratio=args.mlp_ratio, 
                attn_drop=args.attn_drop, 
            ) for _ in range(args.num_hidden_layers)
        ])

        self.img_proj_out = FinalLayer(
            hidden_size=args.hidden_size, 
            patch_size=args.dit_patch_size, 
            out_channels=block_out_channels[-1]
        )

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        reversed_layers_per_block = list(reversed(layers_per_block))
        reversed_cross_attention_dim = list(reversed(cross_attention_dim))
        reversed_transformer_layers_per_block = (list(reversed(transformer_layers_per_block)))

        if args.cross_attn:
            up_block_types = (
                "UpBlock2D", 
                "CrossAttnUpBlock2D", 
                "CrossAttnUpBlock2D", 
                "CrossAttnUpBlock2D", 
            )
        else:
            up_block_types = (
                "UpBlock2D", 
                "UpBlock2D", 
                "UpBlock2D", 
                "UpBlock2D", 
            )

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=reversed_layers_per_block[i] + 1,
                transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=1e-5,
                resnet_act_fn="silu",
                resolution_idx=i,
                resnet_groups=32,
                cross_attention_dim=reversed_cross_attention_dim[i],
                num_attention_heads=reversed_num_attention_heads[i],
                dual_cross_attention=False,
                use_linear_projection=False,
                only_cross_attention=False,
                upcast_attention=False,
                resnet_time_scale_shift="default",
                attention_type="default",
                resnet_skip_time_act=False,
                resnet_out_scale_factor=1.0,
                cross_attention_norm=None,
                attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,
                dropout=args.attn_drop,
            )
            self.up_blocks.append(up_block)

        # out
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0], num_groups=32, eps=1e-5, 
        )

        self.conv_act = get_activation("silu")

        self.img_conv_out = nn.Conv2d(
            block_out_channels[0], args.latent_channels, 3, 1, 1, 
        )

        self.seq_proj_out = nn.Linear(args.hidden_size, args.vocab_size, bias=True)

        if args.output_rescale:
            self.mlp_w = nn.Sequential(
                nn.Linear(args.hidden_size, args.hidden_size, bias=False),
                nn.SiLU(),
                nn.Linear(args.hidden_size, 1, bias=False),
            )

        self.initialize_weights()
        self.initialize_protein_sequence_embedding()

    def initialize_protein_sequence_embedding(self):
        if self.args.esm_embedding == "esm1b" or self.args.esm_embedding == "esm2":
            self.protein_sequence_embedding = ESMEmbed(
                self.args.esm_embedding, self.args.hidden_size, self.args.esm_fixed_embedding
            )
        else:
            self.protein_sequence_embedding = nn.Embedding(
                self.args.vocab_size, self.args.hidden_size, padding_idx=1
            )

    def initialize_weights(self):
        # Initialize token type embeddings
        self.token_type_embeddings.weight.data.normal_(mean=0.0, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.unified_encoder:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        nn.init.normal_(self.img_conv_out.weight, std=0.02)
        nn.init.constant_(self.img_conv_out.bias, 0)


    def forward(self, protein_img, cell_img, protein_seq, time, sqrt_one_minus_alphas_cumprod_t, **kwargs):
        t_emb = self.get_time_embed(sample=protein_img, timestep=time)
        time_embeds = self.time_embedding(t_emb)

        # Tokenize protein sequence
        # Size: B x T x C
        seq_embeds = self.protein_sequence_embedding(protein_seq.squeeze(-1))
        seq_embeds = self.seq_proj_in(seq_embeds)
        seq_embeds = seq_embeds + self.protein_sequence_pos_embed[:, :protein_seq.shape[1]]

        seq_token_type = torch.full(
            size=(seq_embeds.shape[0], 1), fill_value=1, device=seq_embeds.device
        ).long()
        
        seq_embeds = seq_embeds + self.token_type_embeddings(seq_token_type)

        concat_img = torch.cat([protein_img, cell_img], dim=1)
        sample = self.img_conv_in(concat_img)

        # down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=time_embeds,
                    encoder_hidden_states=seq_embeds,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=time_embeds)

            down_block_res_samples += res_samples

        # mid
        sample = self.img_embedding(sample)
        img_token_type = torch.full(
            size=(sample.shape[0], 1), fill_value=0, device=sample.device, 
        ).long()

        img_embeds = sample + self.token_type_embeddings(img_token_type)
        co_embeds = torch.cat([img_embeds, seq_embeds], dim=1)
        x = co_embeds

        for i, block in enumerate(self.unified_encoder):
            x = block(x, time_embeds)

        img_feat = x[:, :img_embeds.shape[1]]
        seq_feat = x[:, img_embeds.shape[1]:]

        img_output = self.img_proj_out(img_feat, time_embeds)
        seq_output = self.seq_proj_out(seq_feat)

        self.img_embed = img_output.clone()
        self.seq_embed = seq_output.clone()

        sample = unpatchify(
            img_output, self.args.block_out_channels[-1], self.args.dit_patch_size
        )

        default_overall_up_factor = 2**self.num_upsamplers
        forward_upsample_size = False
        upsample_size = None

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                forward_upsample_size = True
                break

        # up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample, 
                    temb=time_embeds, 
                    res_hidden_states_tuple=res_samples, 
                    encoder_hidden_states=seq_embeds, 
                    upsample_size=upsample_size, 
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, 
                    temb=time_embeds, 
                    res_hidden_states_tuple=res_samples, 
                    upsample_size=upsample_size, 
                )

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.img_conv_out(sample)

        if self.args.output_rescale:
            if self.args.diffusion_pred_type == "noise":
                scale_shift = self.mlp_w(time_embeds).unsqueeze(-1).unsqueeze(-1)
                logit_bias = torch.logit(sqrt_one_minus_alphas_cumprod_t)
                scale = torch.sigmoid(scale_shift + logit_bias)
                sample = scale * protein_img +  (1 - scale) * sample
            elif self.args.diffusion_pred_type == "xstart":
                scale_shift = self.mlp_w(time_embeds).unsqueeze(-1).unsqueeze(-1)
                logit_bias = torch.logit(sqrt_one_minus_alphas_cumprod_t)
                scale = torch.sigmoid(scale_shift + logit_bias)
                sample = scale * sample + (1 - scale) * protein_img
            else:
                raise ValueError(
                    f"diffusion mode: {self.args.diffusion_pred_type} is not supported"
                )

        return sample, seq_output

    def _set_time_proj(
        self, 
        time_embedding_type: str, 
        block_out_channels: int, 
        flip_sin_to_cos: bool, 
        freq_shift: float, 
        time_embedding_dim: int, 
    ) -> Tuple[int, int]:
        if time_embedding_type == "fourier":
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 2
            if time_embed_dim % 2 != 0:
                raise ValueError(f"`time_embed_dim` should be divisible by 2, but is {time_embed_dim}.")
            self.time_proj = GaussianFourierProjection(
                time_embed_dim // 2, set_W_to_weight=False, log=False, flip_sin_to_cos=flip_sin_to_cos
            )
            timestep_input_dim = time_embed_dim
        elif time_embedding_type == "positional":
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 4

            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = block_out_channels[0]
        else:
            raise ValueError(
                f"{time_embedding_type} does not exist. Please make sure to use one of `fourier` or `positional`."
            )

        return time_embed_dim, timestep_input_dim

    def get_time_embed(
        self, sample: torch.Tensor, timestep: Union[torch.Tensor, float, int]
    ) -> Optional[torch.Tensor]:
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)
        return t_emb
