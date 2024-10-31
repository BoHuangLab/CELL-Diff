import torch
import torch.nn as nn
import numpy as np

from cell_diff.logging import logger
from cell_diff.pipeline.accelerator.dataclasses import ModelOutput
from cell_diff.pipeline.accelerator.trainer import Model
from cell_diff.models.cell_diff.cell_diff_model import CELLDiff
from cell_diff.models.cell_diff.modules.diffusion import create_diffusion

class CELLDiffSeqModel(Model):
    def __init__(self, args, loss_fn=None):
        super().__init__()
        self.args = args

        if args.rank == 0:
            logger.info(self.args)

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
        
        protein_seq_masked = batched_data['protein_seq_masked']
        B = protein_seq_masked.shape[0]
        
        protein_img = torch.full(size=(B, 1, self.args.img_resize, self.args.img_resize), fill_value=-1, device=protein_seq_masked.device).float()
        cell_img = torch.full(size=(B, len(self.args.cell_image.split(',')), self.args.img_resize, self.args.img_resize), fill_value=-1, device=protein_seq_masked.device).float()

        # add noise to protein_img
        time = torch.full((B,), self.diffusion.num_timesteps - 1, device=protein_seq_masked.device)
        protein_img_noisy = torch.randn_like(protein_img)
        sqrt_one_minus_alphas_cumprod_t = self.diffusion.sqrt_one_minus_alphas_cumprod_t(protein_img, time)

        model_time = self.diffusion._scale_timesteps(time)

        if self.args.fp16:
            protein_img_noisy = protein_img_noisy.half()
            cell_img = cell_img.half()
            model_time = model_time.half()

        _, protein_seq_output = self.net(protein_img_noisy, 
                                         cell_img, 
                                         protein_seq_masked, 
                                         model_time, 
                                         sqrt_one_minus_alphas_cumprod_t)

        return protein_seq_output

    def compute_loss(self, model_output, batch_data) -> ModelOutput:
        bs = model_output.shape[0]
        
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

    def masked_sequence_generation(self, protein_seq, protein_seq_mask, progress=True, sampling_strategy='oaardm', order='l2r', temperature=1.0):
        if sampling_strategy == "oaardm":
            loc = torch.where(protein_seq_mask.bool().squeeze())[0].cpu().numpy()

            if order == 'l2r':
                loc = np.sort(loc)
            elif order == 'random':
                np.random.shuffle(loc)
            
            if progress:
                # Lazy import so that we don't depend on tqdm.
                from tqdm.auto import tqdm
                loc = tqdm(loc)
            
            B = protein_seq.shape[0]
            protein_img = torch.randn(size=(B, 1, self.args.img_resize, self.args.img_resize), device=protein_seq.device).float()
            cell_img = torch.full(size=(B, len(self.args.cell_image.split(',')), self.args.img_resize, self.args.img_resize), fill_value=-1, device=protein_seq.device).float()

            with torch.no_grad():
                for i in loc:
                    time = torch.full((B,), self.diffusion.num_timesteps - 1, device=protein_seq.device)
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
