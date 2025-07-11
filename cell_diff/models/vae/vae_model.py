import os
import torch
import torch.nn as nn
from diffusers.models import AutoencoderKL
from .vae_config import VAEConfig
from transformers import PreTrainedModel
from cell_diff.pipeline.utils import VAEOutput
from cell_diff.logging import logger
import random


class VAEModel(PreTrainedModel):
    config_class = VAEConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # Calculate number of downsampling and upsampling layers required
        self.num_down_blocks = config.num_down_blocks  # Input size / latent space size
        self.num_up_blocks = self.num_down_blocks

        # Generate down_block_types and up_block_types based on the required layers
        down_block_types = tuple(["DownEncoderBlock2D"] * self.num_down_blocks)
        up_block_types = tuple(["UpDecoderBlock2D"] * self.num_up_blocks)

        # Initialize AutoencoderKL with custom parameters from config
        self.vae = AutoencoderKL(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=config.vae_block_out_channels,
            latent_channels=config.latent_channels,  # Latent space dimensions
        )

        self.load_pretrained_weights(config, checkpoint_path=config.vae_loadcheck_path)

    def load_pretrained_weights(self, config, checkpoint_path):
        """
        Load pretrained weights from a given state_dict.
        """
        if config.ft or config.infer:
            if config.ft:
                logger.info(f"Finetune from checkpoint: {checkpoint_path}")
            else:
                logger.info(f"Infer from checkpoint: {checkpoint_path}")
                
            if os.path.splitext(checkpoint_path)[1] == '.safetensors':
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

    def encode(self, x):
        """Encodes input into latent space."""
        return self.vae.encode(x).latent_dist

    def decode(self, latents):
        """Decodes latent space into reconstructed input."""
        return self.vae.decode(latents)

    def forward(self, batched_data):
        img_list = [batched_data['protein_img'], batched_data['nucleus_img']]
        if 'ER_img' in batched_data:
            img_list.append(batched_data['ER_img'])
        if 'microtubules_img' in batched_data:
            img_list.append(batched_data['microtubules_img'])
        
        x = random.choice(img_list)

        """Forward pass through the VAE."""

        latent_dist = self.encode(x)
        latents = latent_dist.sample()  # Reparameterization trick
        recon_x = self.decode(latents).sample
        total_loss, recon_loss, kl_loss = self.compute_loss(x, recon_x, latent_dist)

        return VAEOutput(total_loss, recon_loss, kl_loss)

    def compute_loss(self, x, recon_x, latent_dist):
        """Compute reconstruction and KL divergence loss."""
        recon_loss = nn.MSELoss()(recon_x, x)
        kl_loss = -0.5 * torch.mean(1 + latent_dist.logvar - latent_dist.mean.pow(2) - latent_dist.logvar.exp())
        total_loss = self.config.recon_loss_coeff * recon_loss + self.config.kl_loss_coeff * kl_loss
        return total_loss, recon_loss, kl_loss

    def sample(self, num_samples=1, latent_size=64, device="cpu"):
        """
        Generate samples from the latent space.

        Args:
            num_samples (int): Number of samples to generate.
            device (str): Device to perform sampling on.

        Returns:
            torch.Tensor: Generated images.
        """
        # Sample from a standard normal distribution in latent space
        latents = torch.randn((num_samples, self.config.latent_channels, latent_size, latent_size), device=device)  # Shape matches latent dimensions

        # Decode latents to generate images
        with torch.no_grad():
            generated_images = self.decode(latents).sample

        return generated_images
    
    def reconstruct(self, x):
        latent_dist = self.encode(x)
        latents = latent_dist.sample()  # Reparameterization trick
        recon_x = self.decode(latents).sample

        return recon_x