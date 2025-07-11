import torch
from dataclasses import dataclass
from transformers.utils import ModelOutput
from transformers import Trainer
from cell_diff.logging import metric_logger
from typing import Dict, Optional

class CELLDiffTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Compute loss using parent class
        loss, outputs = super().compute_loss(model, inputs, True)
        # metric_logger.log(outputs.log_output)
        # self.log(outputs.log_output)
        
        return (loss, outputs) if return_outputs else loss

@dataclass
class VAEOutput(ModelOutput):
    loss: torch.FloatTensor = None
    recon_loss: torch.FloatTensor = None
    kl_loss: torch.FloatTensor = None

@dataclass
class CELLDiffOutput(ModelOutput):
    loss: torch.FloatTensor = None
    log_output: Optional[Dict] = None