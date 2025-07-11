import torch
import torch.nn as nn
from esm import pretrained


class ESMEmbed(nn.Module):
    def __init__(self, vocab, out_features, fixed_embedding=False):
        super(ESMEmbed, self).__init__()

        # Initialize the model according to the given vocabulary
        self.vocab = vocab
        
        if vocab == "esm1b":
            self.model, _ = pretrained.esm1b_t33_650M_UR50S()
            self.in_features = 1280

        elif vocab == "esm2":
            if out_features == 320:
                self.model, _ = pretrained.esm2_t6_8M_UR50D()
            elif out_features == 480:
                self.model, _ = pretrained.esm2_t12_35M_UR50D()
            elif out_features == 640:
                self.model, _ = pretrained.esm2_t30_150M_UR50D()
            elif out_features == 1280:
                self.model, _ = pretrained.esm2_t33_650M_UR50D()
            elif out_features == 2560:
                self.model, _ = pretrained.esm2_t36_3B_UR50D()
            else:
                self.model, _ = pretrained.esm2_t33_650M_UR50D()
            self.in_features = self.model.embed_dim

        # Set the number of output features and initialize the scaling layer
        self.out_features = out_features
        if self.in_features != self.out_features:
            self.scale_layer = nn.Linear(self.in_features, self.out_features)
        else:
            self.scale_layer = nn.Identity()

        # Determine whether to freeze the model's parameters
        self.fixed_embedding = fixed_embedding
        if self.fixed_embedding:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model = self.model.eval()

    def forward(self, x, **kwargs):
        # If the model's parameters are fixed, use torch.no_grad()
        if self.fixed_embedding:
            with torch.no_grad():
                if self.vocab == "esm1b" or self.vocab == "esm2":
                    # Reduce sequence length dimension, get top layer representation tensor
                    x = self.model(x.squeeze(1), repr_layers=[self.model.num_layers])[
                        "representations"
                    ][self.model.num_layers]
                    # Tensor shape: (batch_size, sequence_length, hidden_size)
                else:
                    # Get top layer representation tensor
                    x = self.model(x, **kwargs)[0]
                    # Tensor shape: (batch_size, sequence_length, hidden_size)
        else:
            if self.vocab == "esm1b" or self.vocab == "esm2":
                # Reduce sequence length dimension, get top layer representation tensor
                x = self.model(x.squeeze(1), repr_layers=[self.model.num_layers])[
                    "representations"
                ][self.model.num_layers]
                # Tensor shape: (batch_size, sequence_length, hidden_size)
            else:
                # Get top layer representation tensor
                x = self.model(x, **kwargs)[0]
                # Tensor shape: (batch_size, sequence_length, hidden_size)

        # Scale the representation tensor if necessary
        if self.out_features != self.in_features:
            x = self.scale_layer(x)
            # Tensor shape: (batch_size, out_features)

        return x

# model = ESMEmbed('esm2', 1280, True)

# seq = torch.tensor([4, 10, 15, 17, 11, 10]).long()
# seq = seq.unsqueeze(0)

# print(model.model.num_layers)

# print(seq.squeeze(1).shape)

# feat = model(seq)
# print(feat.shape)
# print(feat)