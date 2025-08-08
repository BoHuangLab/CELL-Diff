# -*- coding: utf-8 -*-
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from pathlib import Path

from cell_diff.criterions.unidiff import UniDiffCriterions
from cell_diff.models.cell_diff.config import CELLDiffConfig
from cell_diff.models.cell_diff.model import CELLDiffModel
from cell_diff.models.vae.vae_model import VAEModel
from cell_diff.models.vae.vae_config import VAEConfig
from cell_diff.utils.cli_utils import cli

from torchvision.transforms.functional import to_tensor
from cell_diff.data.hpa_data.vocabulary import Alphabet, convert_string_sequence_to_int_index

from torchvision.utils import save_image
from copy import deepcopy
from PIL import Image


def prepare_data(dataroot: Path, vae: VAEModel):
    protein_img = to_tensor(Image.open(os.path.join(dataroot / "protein.png")).convert("L")).to(vae.device)
    nucleus_img = to_tensor(Image.open(os.path.join(dataroot / "nucleus.png")).convert("L")).to(vae.device)

    protein_img = protein_img * 2 - 1
    nucleus_img = nucleus_img * 2 - 1

    with torch.no_grad():
        protein_img_latent = vae.encode(protein_img.unsqueeze(0)).sample()
        nucleus_img_latent = vae.encode(nucleus_img.unsqueeze(0)).sample()

        cell_img_latent = nucleus_img_latent

    return protein_img_latent, cell_img_latent

def colorize_image(tensor, color):
    colored_image = torch.zeros((3, tensor.size(0), tensor.size(1)), dtype=tensor.dtype, device=tensor.device)

    if color == 'blue':
        colored_image[2] = tensor  # Set blue channel
    elif color == 'red':
        colored_image[0] = tensor  # Set red channel
    elif color == 'green':
        colored_image[1] = tensor  # Set green channel
    elif color == 'yellow':
        colored_image[0] = tensor  # Set red and green channels to get yellow
        colored_image[1] = tensor
    return colored_image

def save_colored_image(tensor, filename, color):
    colored_tensor = colorize_image(tensor, color)
    save_image(colored_tensor, filename, normalize=True, value_range=(0, 1))

@cli(CELLDiffConfig)
def main(args) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    vae_args = deepcopy(args)
    vae_args.infer = True

    vae = VAEModel(config=VAEConfig(**vars(vae_args)))

    for param in vae.parameters():
        param.requires_grad = False
    vae.to(device)
    vae.eval()

    model = CELLDiffModel(config=CELLDiffConfig(**vars(args)), loss_fn=UniDiffCriterions)

    model.to(device)
    model.eval()

    protein_img_latent, cell_img_latent = prepare_data(Path(args.image_path), vae)

    vocab = Alphabet()

    num_aas = args.num_aas
    num_gen = 10

    masked_seq = "M" + "<mask>" * num_aas

    protein_seq = convert_string_sequence_to_int_index(vocab, masked_seq)
    protein_seq = torch.LongTensor(protein_seq).unsqueeze(0).to(device)

    print(vocab.untokenize(protein_seq.squeeze()[1:-1]))
    protein_seq_mask = (protein_seq == vocab.mask_idx)

    for i in range(num_gen):
        with torch.no_grad():
            sample = model.image_to_sequence(
                protein_seq, protein_seq_mask, protein_img_latent,
                cell_img_latent, order='random', temperature=1.0,
                progress=False,
            )

        gen_seq = vocab.untokenize(sample.squeeze()[1:-1])
        gen_sig = gen_seq[-num_aas:]
        print(f"Sample {i+1}:")
        print(gen_sig)


if __name__ == "__main__":
    main()
