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
    nucleus_img = to_tensor(Image.open(os.path.join(dataroot / "nucleus.png")).convert("L")).to(vae.device)
    nucleus_img = nucleus_img * 2 - 1
    nucleus_img = nucleus_img.unsqueeze(0)

    with torch.no_grad():
        nucleus_img_latent = vae.encode(nucleus_img).sample()
    
    cell_img_latent = nucleus_img_latent

    return cell_img_latent, nucleus_img

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

    cell_img_latent, nucleus_img = prepare_data(Path(args.image_path), vae)

    test_sequence = args.test_sequence

    vocab = Alphabet()
    test_sequence_token = convert_string_sequence_to_int_index(vocab, test_sequence)
    test_sequence_token = torch.LongTensor(test_sequence_token).unsqueeze(0).to(device)

    print("Generating protein image from sequence: ")
    print(test_sequence)

    with torch.no_grad():
        sample = model.sequence_to_image(test_sequence_token, cell_img_latent, sampling_strategy="ddim")
        sample = vae.decode(sample).sample

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    save_colored_image((sample.squeeze() + 1) / 2, output_dir / 'generated_protein_img.png', 'green')
    cat_img = torch.cat([torch.full_like(nucleus_img, -1), sample, nucleus_img], dim=1)
    save_image(cat_img, output_dir / 'generated_img.png', normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    main()
