import torch
import torch.nn as nn

class UniDiffCriterions(nn.Module):
    def __init__(self, args, reduction="none") -> None:
        super().__init__()
        self.sequence_loss = nn.CrossEntropyLoss(reduction=reduction)

        self.args = args
        self.sequence_loss_coeff = self.args.sequence_loss_coeff
        self.image_loss_coeff = self.args.image_loss_coeff

    def forward(self, batch_data, model_output):
        with torch.no_grad():
            protein_seq_mask = batch_data["protein_seq_mask"].squeeze(-1).bool()
            protein_seq = batch_data["protein_seq"][protein_seq_mask]
            zm_label = batch_data['zm_label'].squeeze(-1).bool()

            mask_num = torch.sum(protein_seq_mask.float(), dim=1, keepdim=True)
            seq_loss_weights = torch.ones_like(protein_seq_mask, dtype=torch.float32)
            seq_loss_weights = seq_loss_weights / mask_num
            seq_loss_weights = seq_loss_weights * protein_seq_mask.float()
            seq_loss_weights = seq_loss_weights[protein_seq_mask.bool()]
            
        _, protein_seq_output, image_loss = model_output

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

        if zm_label.all():
            sequence_loss = torch.zeros(1, device=sequence_loss.device).sum()
        else:
            sequence_loss = sequence_loss.sum() / (~zm_label).float().sum()
        
        # Protein image loss
        if zm_label.any():
            image_loss = image_loss[zm_label].mean() * self.image_loss_coeff
        else:
            image_loss = torch.zeros(1, device=image_loss.device).sum()
        
        loss = sequence_loss + image_loss

        log_loss = {'Sequence loss': sequence_loss.item() / self.sequence_loss_coeff, 
                    'Image loss': image_loss.item() / self.image_loss_coeff}
        
        return loss, log_loss