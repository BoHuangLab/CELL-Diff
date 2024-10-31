import torch

import torch.distributed as dist
import torch.nn as nn


class UniDiffCriterions(nn.Module):
    def __init__(self, args, reduction="mean") -> None:
        super().__init__()

        self.sequence_loss = nn.CrossEntropyLoss(reduction='none')
        self.image_loss = nn.MSELoss(reduction=reduction)

        self.args = args
        self.sequence_loss_coeff = self.args.sequence_loss_coeff
        self.image_loss_coeff = self.args.image_loss_coeff

    def forward(self, batch_data, model_output):
        with torch.no_grad():
            protein_seq_mask = batch_data["protein_seq_mask"].squeeze(-1).bool()            
            protein_seq = batch_data["protein_seq"][protein_seq_mask]
            protein_img = batch_data["protein_img"]
            
            mask_num = torch.sum(protein_seq_mask.float(), dim=1, keepdim=True)
            seq_loss_weights = torch.ones_like(protein_seq_mask, dtype=torch.float32)
            seq_loss_weights = seq_loss_weights / mask_num
            seq_loss_weights = seq_loss_weights * protein_seq_mask.float()
            seq_loss_weights = seq_loss_weights[protein_seq_mask.bool()]
            
        protein_img_output, protein_seq_output, image_loss = model_output

        # Protein sequence loss
        protein_seq_output = protein_seq_output[protein_seq_mask]
        sequence_loss = (
            self.sequence_loss(
                protein_seq_output.to(torch.float32).view(-1, protein_seq_output.size(-1)),
                protein_seq.view(-1), 
            )
            * self.sequence_loss_coeff
        )
        sequence_loss = sequence_loss * seq_loss_weights
        sequence_loss = sequence_loss.sum() / protein_seq_mask.shape[0]
        
        # Protein image loss
        image_loss = image_loss.mean() * self.image_loss_coeff
        
        loss = sequence_loss + image_loss

        log_loss = {'Sequence loss': sequence_loss / self.sequence_loss_coeff, 
                    'Image loss': image_loss / self.image_loss_coeff}
        
        return loss, log_loss