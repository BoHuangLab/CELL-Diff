import torch
import torch.nn as nn
import math


class GaussianDiffusion:
    def __init__(self, 
                 num_timesteps, 
                 ddpm_schedule, 
                 ddpm_beta_start, 
                 ddpm_beta_end):
        super().__init__()

        self.num_timesteps = num_timesteps
        self.ddpm_schedule = ddpm_schedule
        assert ddpm_schedule in ["linear", "quadratic", "sigmoid", "cosine"]
        (
            self.betas, 
            self.alphas, 
            self.alphas_cumprod, 
            self.one_minus_alphas_cumprod, 
            self.sqrt_alphas_cumprod,
            self.sqrt_one_minus_alphas_cumprod,
        ) = self._beta_schedule(
            num_timesteps,
            ddpm_beta_start,
            ddpm_beta_end,
            ddpm_schedule,
        )

    def _beta_schedule(self, num_timesteps, beta_start, beta_end, schedule_type="linear"):
        if schedule_type == "linear":
            beta_list = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == "quadratic":
            beta_list = (
                torch.linspace(beta_start**0.5, beta_end**0.5, num_timesteps) ** 2
            )
        elif schedule_type == "sigmoid":
            betas = torch.linspace(-6, 6, num_timesteps)
            beta_list = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
        elif schedule_type == "cosine":
            s = 0.008
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps)
            alphas_cumprod = (
                torch.cos(((x / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            )
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            beta_list = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise NotImplementedError

        alphas = 1 - beta_list
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        one_minus_alphas_cumprod = 1.0 - alphas_cumprod
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(one_minus_alphas_cumprod)
        return beta_list, alphas, alphas_cumprod, one_minus_alphas_cumprod, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod

    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def _noise_sample(self, x_start, t):
        noise = torch.randn_like(x_start) * 1.0

        sqrt_alphas_cumprod_t = self._extract(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def _ssnr(self, x, t):
        return self._extract(self.alphas_cumprod, t, x.shape)

    def _ddpm_mean_var(self, denoised, x, t):

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x.shape)
        beta_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        alpha_t = self._extract(self.alphas, t, x.shape)
        one_minus_alphas_cumprod_t = self._extract(self.one_minus_alphas_cumprod, t, x.shape)
        alphas_cumprod_t = self._extract(self.alphas_cumprod, t, x.shape)
        sqrt_alpha_t = torch.sqrt(alpha_t)

        coeff1 = sqrt_alphas_cumprod_t * beta_t / (one_minus_alphas_cumprod_t * sqrt_alpha_t)
        coeff2 = (alpha_t - alphas_cumprod_t) / (one_minus_alphas_cumprod_t * sqrt_alpha_t)

        mean = coeff1 * denoised + coeff2 * x

        var = (alpha_t - alphas_cumprod_t) * beta_t / (alpha_t * one_minus_alphas_cumprod_t)

        return mean, var

    def _ddim_mean_var(self, denoised, x, t, diff_skip_ratio=1):

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x.shape)
        beta_t = self._extract(self.betas, t, x.shape) * diff_skip_ratio
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        alpha_t = self._extract(self.alphas, t, x.shape)
        one_minus_alphas_cumprod_t = self._extract(self.one_minus_alphas_cumprod, t, x.shape)
        alphas_cumprod_t = self._extract(self.alphas_cumprod, t, x.shape)
        sqrt_alpha_t = torch.sqrt(alpha_t)

        coeff1 = (sqrt_alphas_cumprod_t / torch.sqrt(1 - beta_t)) * (1 - torch.sqrt(1 - beta_t / one_minus_alphas_cumprod_t))
        coeff1[t == 0] = 1.0
        coeff2 = (1 / torch.sqrt(1 - beta_t)) * torch.sqrt(1 - beta_t / (1 - alphas_cumprod_t))
        coeff2[t == 0] = 0.0

        mean = coeff1 * denoised + coeff2 * x

        var = torch.zeros_like(mean)

        return mean, var


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(
        self,
        dim,
        max_period=10000,
    ):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.dummy = nn.Parameter(
            torch.empty(0, dtype=torch.float), requires_grad=False
        )  # to detect fp16

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(self.max_period) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None].float() * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        embeddings = embeddings.to(self.dummy.dtype)
        return embeddings

class TimeStepEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super(TimeStepEncoder, self).__init__()

        self.time_proj = SinusoidalPositionEmbeddings(embedding_dim)

        self.time_embedding = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, timesteps):
        t_emb = self.time_proj(timesteps)
        t_emb = self.time_embedding(t_emb)
        return t_emb