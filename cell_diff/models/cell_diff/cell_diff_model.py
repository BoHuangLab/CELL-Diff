import torch
import torch.nn as nn
import numpy as np

from cell_diff.logging import logger
from cell_diff.pipeline.accelerator.dataclasses import ModelOutput
from cell_diff.pipeline.accelerator.trainer import Model
from timm.models.vision_transformer import PatchEmbed

from cell_diff.models.cell_diff.modules.protein_sequence_embedding import ESMEmbed

from cell_diff.models.cell_diff.modules.diffusion import TimeStepEncoder, create_diffusion
from cell_diff.models.cell_diff.modules.transformer import TransformerBlock, unpatchify, FinalLayer
from cell_diff.models.cell_diff.modules.positional_embedding import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed
from cell_diff.models.cell_diff.modules.simple_diffusion import ResnetBlock, Upsample, Downsample, ResAttBlock
import random


class CELLDiffModel(Model):
    def __init__(self, args, loss_fn=None):
        super().__init__()
        self.args = args

        self.loss = loss_fn(args)

        self.net = CELLDiff(args)
        if self.args.diffusion_pred_type == 'xstart':
            predict_xstart = True
        elif self.args.diffusion_pred_type == 'noise':
            predict_xstart = False
        self.diffusion = create_diffusion(timestep_respacing=args.timestep_respacing, 
                                          noise_schedule=args.ddpm_schedule, 
                                          learn_sigma=False, 
                                          image_d=args.img_resize, 
                                          predict_xstart=predict_xstart)

        self.load_pretrained_weights(args, checkpoint_path=args.loadcheck_path)

    def load_pretrained_weights(self, args, checkpoint_path):
        """
        Load pretrained weights from a given state_dict.
        """
        if args.ft or args.infer:
            if args.ft:
                logger.info(f"Finetune from checkpoint: {checkpoint_path}")
            else:
                logger.info(f"Infer from checkpoint: {checkpoint_path}")
                
            checkpoints_state = torch.load(checkpoint_path, map_location="cpu")

            if "model" in checkpoints_state:
                checkpoints_state = checkpoints_state["model"]
            elif "module" in checkpoints_state:
                checkpoints_state = checkpoints_state["module"]

            model_state_dict = self.state_dict()
            filtered_state_dict = {k: v for k, v in checkpoints_state.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}

            IncompatibleKeys = self.load_state_dict(filtered_state_dict, strict=False)
            # IncompatibleKeys = self.load_state_dict(checkpoints_state, strict=False)
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

    def forward(self, batched_data, **kwargs):
        # batched_data = {"protein_img": protein_img, 
        #                 "nucleus_img": nucleus_img, 
        #                 "microtubules_img": microtubules_img, 
        #                 "ER_img": ER_img, 
        #                 "cell_line": cell_line, 
        #                 "rna_expression": rna_expression, 
        #                 "protein_seq": protein_seq, 
        #                 "protein_seq_masked": protein_seq_masked, 
        #                 "protein_seq_mask": protein_seq_mask}        

        # protein_img = batched_data['protein_img']
        # nucleus_img = batched_data['nucleus_img']
        # microtubules_img = batched_data['microtubules_img']
        # rna_expression = batched_data['rna_expression']
        # protein_seq = batched_data['protein_seq']
        # protein_seq_masked = batched_data['protein_seq_masked']
        # protein_seq_mask = batched_data['protein_seq_mask']
        
        protein_img = batched_data['protein_img']
        protein_seq_masked = batched_data['protein_seq_masked']
        zm_label = batched_data['zm_label'].squeeze(-1).bool()
                
        if self.args.cell_image == 'nucl':
            cell_img = batched_data['nucleus_img']
        elif self.args.cell_image == 'nucl,er':
            ER_img = batched_data['ER_img'] if random.uniform(0, 1) <= self.args.cell_image_ratio else torch.full_like(batched_data['ER_img'], -1)
            cell_img = torch.cat([batched_data['nucleus_img'], ER_img], dim=1)
        elif self.args.cell_image == 'nucl,mt':
            microtubules_img = batched_data['microtubules_img'] if random.uniform(0, 1) <= self.args.cell_image_ratio else torch.full_like(batched_data['microtubules_img'], -1)
            cell_img = torch.cat([batched_data['nucleus_img'], microtubules_img], dim=1)
        elif self.args.cell_image == 'nucl,er,mt':
            ER_img = batched_data['ER_img'] if random.uniform(0, 1) <= self.args.cell_image_ratio else torch.full_like(batched_data['ER_img'], -1)
            microtubules_img = batched_data['microtubules_img'] if random.uniform(0, 1) <= self.args.cell_image_ratio else torch.full_like(batched_data['microtubules_img'], -1)
            cell_img = torch.cat([batched_data['nucleus_img'], ER_img, microtubules_img], dim=1)
        else:
            raise ValueError(f"Cell image type: {self.args.cell_image} is not supported")
        
        # add noise to protein_img
        time = torch.randint(0, 
                             self.diffusion.num_timesteps, 
                             (protein_img.shape[0],), 
                             device=protein_img.device)
        time[zm_label] = self.diffusion.num_timesteps - 1

        noise = torch.randn_like(protein_img)
        protein_img_noisy = self.diffusion.q_sample(protein_img, time, noise)
        sqrt_one_minus_alphas_cumprod_t = self.diffusion.sqrt_one_minus_alphas_cumprod_t(protein_img, time)

        model_time = self.diffusion._scale_timesteps(time)

        if self.args.fp16:
            protein_img_noisy = protein_img_noisy.half()
            cell_img = cell_img.half()
            model_time = model_time.half()

        protein_img_output, protein_seq_output = self.net(protein_img_noisy, 
                                                          cell_img, 
                                                          protein_seq_masked, 
                                                          model_time, 
                                                          sqrt_one_minus_alphas_cumprod_t)

        diff_loss_dict = self.diffusion.training_losses(protein_img_output.to(torch.float32), 
                                                        protein_img_noisy.to(torch.float32), 
                                                        protein_img.to(torch.float32), 
                                                        time, 
                                                        noise)

        return protein_img_output, protein_seq_output, diff_loss_dict['loss']

    def compute_loss(self, model_output, batch_data) -> ModelOutput:
        protein_img_output = model_output[0]
        bs = protein_img_output.shape[0]
        
        output = self.loss(batch_data, model_output)
        loss = output[0]
        log_loss = output[1]
        
        return ModelOutput(loss=loss, log_output=log_loss, num_examples=bs)

    def config_optimizer(self):
        """
        Return the optimizer and learning rate scheduler for this model.

        Returns:
            tuple[Optimizer, LRScheduler]:
        """
        pass

    def sequence_to_image(self, protein_seq, cell_img, progress=True, sampling_strategy="ddpm"):
        protein_img = torch.randn(cell_img.shape[0], 1, cell_img.shape[2], cell_img.shape[3]).to(cell_img.device)
        indices = list(range(self.diffusion.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            time = torch.tensor([i] * cell_img.shape[0], device=cell_img.device)
            sqrt_one_minus_alphas_cumprod_t = self.diffusion.sqrt_one_minus_alphas_cumprod_t(protein_img, time)
            with torch.no_grad():
                model_time = self.diffusion._scale_timesteps(time)
                protein_img_output = self.net(protein_img, 
                                              cell_img, 
                                              protein_seq, 
                                              model_time, 
                                              sqrt_one_minus_alphas_cumprod_t)[0]
                if sampling_strategy == "ddpm":
                    out = self.diffusion.p_sample(
                        protein_img_output, 
                        protein_img, 
                        time, 
                        clip_denoised=True, 
                    )
                    protein_img = out["sample"]
                elif sampling_strategy == "ddim":
                    out = self.diffusion.ddim_sample(
                        protein_img_output,
                        protein_img,
                        time,
                        clip_denoised=True, 
                    )
                    protein_img = out["sample"]
        
        return protein_img

    def image_to_sequece(self, protein_seq, protein_seq_mask, protein_img, cell_img, progress=True, sampling_strategy='oaardm', order='l2r', temperature=1.0):
        if sampling_strategy == "oaardm":
            return self.oaardm_sample(protein_seq, protein_seq_mask, protein_img, cell_img, progress, order, temperature)

    def oaardm_sample(self, protein_seq, protein_seq_mask, protein_img, cell_img, progress=True, order='l2r', temperature=1.0):
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
                sqrt_one_minus_alphas_cumprod_t = self.diffusion.sqrt_one_minus_alphas_cumprod_t(protein_img, time)

                model_time = self.diffusion._scale_timesteps(time)

                protein_seq_output = self.net(protein_img, 
                                              cell_img, 
                                              protein_seq, 
                                              model_time, 
                                              sqrt_one_minus_alphas_cumprod_t)[1]
                
                p = protein_seq_output[:, i, 4:4+20] # sample at location i (random), dont let it predict non-standard AA
                p = torch.nn.functional.softmax(p / temperature, dim=1) # softmax over categorical probs
                p_sample = torch.multinomial(p, num_samples=1)
                protein_seq[:, i] = p_sample.squeeze() + 4

        return protein_seq


class CELLDiff(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # Use fixed sin-cos embedding:
        self.protein_sequence_pos_embed = nn.Parameter(torch.zeros(1, args.max_protein_sequence_len, args.embed_dim), requires_grad=False)

        # Potential condition
        self.cell_line_embedding = None
        
        # Use ConvNet to process image
        self.inconv = nn.Conv2d(args.img_in_chans*(1 + len(self.args.cell_image.split(','))), args.dims[0], 3, 1, 1)

        self.downs = {}
        for i_level in range(len(args.num_res_block)):
            for i_block in range(args.num_res_block[i_level]):
                if self.args.cross_attn:
                    patch_size = args.img_resize // 2**i_level // args.res_att_num_patches                
                    self.downs[str(i_level)+str(i_block)] = ResAttBlock(args.dims[i_level], 
                                                                        args.dims[i_level], 
                                                                        patch_size, 
                                                                        args.embed_dim, 
                                                                        args.num_heads, 
                                                                        attn_drop=args.attn_drop)
                else:
                    self.downs[str(i_level)+str(i_block)] = ResnetBlock(args.dims[i_level], 
                                                                        args.dims[i_level], 
                                                                        args.embed_dim)

            self.downs['down'+str(i_level)] = Downsample(args.dims[i_level], args.dims[i_level+1])
        self.downs = nn.ModuleDict(self.downs)

        self.ups = {}
        for i_level in reversed(range(len(args.num_res_block))):
            self.ups['up'+str(i_level)] = Upsample(args.dims[i_level+1], args.dims[i_level])
            for i_block in range(args.num_res_block[i_level]):
                if self.args.cross_attn:
                    patch_size = args.img_resize // 2**i_level // args.res_att_num_patches
                    self.ups[str(i_level)+str(i_block)] = ResAttBlock(args.dims[i_level]*2, 
                                                                      args.dims[i_level], 
                                                                      patch_size, 
                                                                      args.embed_dim, 
                                                                      args.num_heads, 
                                                                      attn_drop=args.attn_drop)
                else:
                    self.ups[str(i_level)+str(i_block)] = ResnetBlock(args.dims[i_level]*2, 
                                                                        args.dims[i_level], 
                                                                        args.embed_dim)                    

        self.ups = nn.ModuleDict(self.ups)

        self.outconv = nn.Conv2d(args.dims[0], args.img_in_chans, 3, 1, 1)

        self.img_embedding = PatchEmbed(img_size=args.img_resize // (2 ** len(args.num_res_block)), 
                                        patch_size=args.dit_patch_size, 
                                        in_chans=args.dims[-1], 
                                        embed_dim=args.embed_dim, 
                                        bias=True)

        # Use fixed sin-cos embedding:
        self.img_pos_embed = nn.Parameter(torch.zeros(1, self.img_embedding.num_patches, args.embed_dim), requires_grad=False)
        
        self.t_embedding = TimeStepEncoder(embedding_dim=args.embed_dim)

        self.token_type_embeddings = nn.Embedding(2, args.embed_dim)

        self.unified_encoder = nn.ModuleList([
            TransformerBlock(args.embed_dim, 
                             args.num_heads, 
                             mlp_ratio=args.mlp_ratio, 
                             attn_drop=args.attn_drop) for _ in range(args.depth)
        ])

        self.img_proj_out = FinalLayer(hidden_size=args.embed_dim, 
                                       patch_size=args.dit_patch_size, 
                                       out_channels=args.dims[-1])

        self.protein_sequence_proj_out = nn.Linear(args.embed_dim, args.num_residues, bias=True)

        self.mlp_w = nn.Sequential(
            nn.Linear(args.embed_dim, args.embed_dim, bias=False),
            nn.SiLU(),
            nn.Linear(args.embed_dim, 1, bias=False),
        )

        self.initialize_weights()
        self.initialize_protein_sequence_embedding()

    def initialize_protein_sequence_embedding(self):
        if self.args.esm_embedding == "esm1b" or self.args.esm_embedding == "esm2":
            self.protein_sequence_embedding = ESMEmbed(self.args.esm_embedding, self.args.embed_dim, self.args.esm_fixed_embedding)
        else:
            self.protein_sequence_embedding = nn.Embedding(self.args.num_residues, self.args.embed_dim, padding_idx=1)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        img_pos_embed = get_2d_sincos_pos_embed(self.img_pos_embed.shape[-1], int(self.img_embedding.num_patches ** 0.5))
        self.img_pos_embed.data.copy_(torch.from_numpy(img_pos_embed).float().unsqueeze(0))

        protein_sequence_pos_embed = get_1d_sincos_pos_embed(self.protein_sequence_pos_embed.shape[-1], self.args.max_protein_sequence_len)
        self.protein_sequence_pos_embed.data.copy_(torch.from_numpy(protein_sequence_pos_embed).float().unsqueeze(0))

        # Initialize token type embeddings
        self.token_type_embeddings.weight.data.normal_(mean=0.0, std=0.02)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.img_embedding.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.img_embedding.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedding.time_embedding[0].weight, std=0.02)
        nn.init.normal_(self.t_embedding.time_embedding[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.unified_encoder:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.img_proj_out.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.img_proj_out.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.img_proj_out.linear.weight, 0)
        nn.init.constant_(self.img_proj_out.linear.bias, 0)
        
        nn.init.constant_(self.protein_sequence_proj_out.weight, 0)
        nn.init.constant_(self.protein_sequence_proj_out.bias, 0)

    def forward(
        self, 
        protein_img, 
        cell_img, 
        protein_seq, 
        time, 
        sqrt_one_minus_alphas_cumprod_t, 
        **kwargs,
    ):
        time_embeds = self.t_embedding(time)

        concat_img = torch.cat([protein_img, cell_img], dim=1)        
        concat_img = self.inconv(concat_img)
        
        # Tokenize protein sequence
        # Size: B x T x C
        
        protein_seq_embeds = self.protein_sequence_embedding(protein_seq.squeeze(-1))
        protein_seq_embeds = protein_seq_embeds + self.protein_sequence_pos_embed[:, : protein_seq_embeds.shape[1]]

        protein_seq_token_type = torch.full(size=(protein_seq_embeds.shape[0], 1), 
                                            fill_value=1, 
                                            device=protein_seq_embeds.device).long()
        
        protein_seq_embeds = protein_seq_embeds + self.token_type_embeddings(protein_seq_token_type)

        # image downsample
        img_skips = []

        for i_level in range(len(self.args.num_res_block)):
            for i_block in range(self.args.num_res_block[i_level]):
                if self.args.cross_attn:
                    concat_img = self.downs[str(i_level)+str(i_block)](concat_img, time_embeds, protein_seq_embeds)
                else:
                    concat_img = self.downs[str(i_level)+str(i_block)](concat_img, time_embeds)
                img_skips.append(concat_img)
            concat_img = self.downs['down'+str(i_level)](concat_img)

        # Tokenize protein image
        # Size: B x T x C

        concat_img_embeds = self.img_embedding(concat_img) + self.img_pos_embed

        concat_img_token_type = torch.full(size=(concat_img_embeds.shape[0], 1), 
                                           fill_value=0, 
                                           device=concat_img_embeds.device).long()

        concat_img_embeds = concat_img_embeds + self.token_type_embeddings(concat_img_token_type)

        concat_img_cond_embeds = concat_img_embeds
        
        co_embeds = torch.cat([concat_img_cond_embeds, protein_seq_embeds], dim=1)
        
        x = co_embeds
        
        time_embed_mask = torch.ones(size=(x.shape[0], x.shape[1]), device=x.device).bool()

        if self.args.transformer_type == 'uvit':
            skips = []
            
            for i, block in enumerate(self.unified_encoder):
                if i < len(self.unified_encoder) // 2:
                    protein_img_feat = x.clone()
                    protein_img_feat[:, self.img_embedding.num_patches :] = 0
                    skips.append(protein_img_feat)
                
                if i > len(self.unified_encoder) // 2:
                    x = x + skips.pop()

                x = block(x, time_embeds, time_embed_mask)
            x = x + skips.pop()

        elif self.args.transformer_type == 'vit':
            for i, block in enumerate(self.unified_encoder):
                x = block(x, time_embeds, time_embed_mask)

        protein_img_feat = x[:, : self.img_embedding.num_patches]
        protein_seq_feat = x[:, self.img_embedding.num_patches :]
        
        protein_img_output = self.img_proj_out(protein_img_feat, time_embeds)
        protein_seq_output = self.protein_sequence_proj_out(protein_seq_feat)

        protein_img_output = unpatchify(protein_img_output, self.args.dims[-1], self.args.dit_patch_size)

        # image upsample
        for i_level in reversed(range(len(self.args.num_res_block))):
            protein_img_output = self.ups['up'+str(i_level)](protein_img_output)
            for i_block in range(self.args.num_res_block[i_level]):
                protein_img_output = torch.cat((protein_img_output, img_skips.pop()), dim = 1)
                if self.args.cross_attn:
                    protein_img_output = self.ups[str(i_level)+str(i_block)](protein_img_output, time_embeds, protein_seq_feat)
                else:
                    protein_img_output = self.ups[str(i_level)+str(i_block)](protein_img_output, time_embeds)

        protein_img_output = self.outconv(protein_img_output)

        if self.args.diffusion_pred_type == "noise":
            scale_shift = self.mlp_w(time_embeds).unsqueeze(-1).unsqueeze(-1)
            logit_bias = torch.logit(sqrt_one_minus_alphas_cumprod_t)
            scale = torch.sigmoid(scale_shift + logit_bias)
            protein_img_output = scale * protein_img +  (1 - scale) * protein_img_output
        elif self.args.diffusion_pred_type == "xstart":
            scale_shift = self.mlp_w(time_embeds).unsqueeze(-1).unsqueeze(-1)
            logit_bias = torch.logit(sqrt_one_minus_alphas_cumprod_t)
            scale = torch.sigmoid(scale_shift + logit_bias)
            protein_img_output = scale * protein_img_output + (1 - scale) * protein_img
        else:
            raise ValueError(
                f"diffusion mode: {self.args.diffusion_pred_type} is not supported"
            )

        return protein_img_output, protein_seq_output
