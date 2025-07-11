# -*- coding: utf-8 -*-
import torch
from cell_diff.logging import logger


class FP16Scaler(object):
    def __init__(
        self,
        init_scale: int,
        scale_factor: float = 2.0,
        scale_interval: int = 1000,
        enabled: bool = False,
    ) -> None:
        self.enabled = enabled
        self.scale = init_scale
        self.scale_factor = scale_factor
        self.since_last_scale_up = 0
        self.scale_interval = scale_interval

    def check_grad_overflow(self, params) -> bool:
        for p in params:
            if p.grad is None:
                continue

            grad_norm = p.grad.data.norm()
            if torch.isinf(grad_norm) or torch.isnan(grad_norm):
                return True

        return False

    def backward(self, loss):
        if self.enabled:
            scaled_loss = loss * self.scale
        else:
            scaled_loss = loss
        scaled_loss.backward()

    def unscale_and_clip_grad(self, params, clip_grad_norm: float):
        for p in params:
            if p.grad is not None:
                p.grad.data = p.grad.data.float()
                p.grad.data /= self.scale

                if clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(p, clip_grad_norm)

    def step(self, model, optimizer, clip_grad_norm: float = 0.0):
        params = model.parameters()
        if self.enabled:
            if self.check_grad_overflow(params):
                self.scale /= self.scale_factor
                logger.info(
                    f"Gradient overflow detected, reducing scale to {self.scale}"
                )
                self.since_last_scale_up = 0
                # Skip optimizer step
            else:
                self.unscale_and_clip_grad(params, clip_grad_norm)
                optimizer.step()

                self.since_last_scale_up += 1
                if (
                    self.since_last_scale_up >= self.scale_interval
                    and self.scale < 2**15
                ):
                    self.scale *= self.scale_factor
                    self.since_last_scale_up = 0
        else:
            self.unscale_and_clip_grad(params, clip_grad_norm)
            optimizer.step()
