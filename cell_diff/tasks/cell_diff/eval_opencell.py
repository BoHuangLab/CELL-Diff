# -*- coding: utf-8 -*-
import os
import sys
import shutil
import glob
import torch
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from pathlib import Path

from cell_diff.criterions.unidiff import UniDiffCriterions
from cell_diff.data.opencell_data.dataset import OpenCellLMDBDataset
from cell_diff.models.cell_diff.config import CELLDiffConfig
from cell_diff.models.cell_diff.model import CELLDiffModel
from cell_diff.models.vae.vae_model import VAEModel
from cell_diff.models.vae.vae_config import VAEConfig
from cell_diff.utils.cli_utils import cli
from cell_diff.logging import logger

from cell_diff.metrics.iou import compute_iou, binarize_img
from cell_diff.metrics.msfr import compute_msf_resolution
from torchvision.utils import save_image
from copy import deepcopy



def put_imgs_to_new_file(new_file_path, img_files):
    
    os.makedirs(new_file_path, exist_ok=True)
    
    for img_file in img_files:
        new_name = img_file.split('/')[-2] + '_' + img_file.split('/')[-1]
        
        new_file = os.path.join(new_file_path, new_name)        
        shutil.copy(img_file, new_file)

def colorize_image(tensor, color):
    # Create a zero tensor with the same size as the input tensor but with three channels
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
    if not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda"

    vae_args = deepcopy(args)
    vae_args.infer = True

    vae = VAEModel(config=VAEConfig(**vars(vae_args)))

    for param in vae.parameters():
        param.requires_grad = False
    vae.to(device)
    vae.eval()

    valset = OpenCellLMDBDataset(args, split_key=args.split_key, vae=vae)
    vocab = valset.vocab

    model = CELLDiffModel(config=CELLDiffConfig(**vars(args)), loss_fn=UniDiffCriterions)

    model.to(device)
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir = output_dir / args.split_key

    ious = []
    # msfs_real = []
    # msfs_gen = []

    for i, data in enumerate(valset):                
        protein_seq = data['protein_seq'].unsqueeze(0).to(device)

        if protein_seq.shape[1] > args.max_protein_sequence_len:
            continue

        with torch.no_grad():
            protein_seq = data['protein_seq'].unsqueeze(0).to(device)

            protein_img = data['protein_img'].unsqueeze(0).to(device)
            nucleus_img = data['nucleus_img'].unsqueeze(0).to(device)
            nucleus_img_latent = vae.encode(nucleus_img).sample()
            cell_img_latent = nucleus_img_latent
        
        logger.info(data['prot_id'])

        save_file = output_dir / ('{:04d}_'.format(i+1) + data['prot_id'])
        save_file.mkdir(parents=True, exist_ok=True)
        
        real_protein_seq = vocab.untokenize(protein_seq.squeeze()[1:-1])
        logger.info(real_protein_seq)
        
        real_img = torch.cat([torch.full_like(protein_img, -1), protein_img, nucleus_img], dim=1)
        save_image(real_img, save_file / 'real_img.png', normalize=True, value_range=(-1, 1))
        save_image(nucleus_img, save_file / 'real_nucleus_img.png', normalize=True, value_range=(-1, 1))
        save_image(protein_img, save_file / 'real_protein_img.png', normalize=True, value_range=(-1, 1))

        sample = model.sequence_to_image(
            protein_seq, cell_img_latent, sampling_strategy="ddim", progress=False, 
        )
        sample = vae.decode(sample).sample
        
        pred_img = torch.cat([torch.full_like(protein_img, -1), sample, nucleus_img], dim=1)

        save_image(pred_img, save_file / 'generated_img.png', normalize=True, value_range=(-1, 1))
        save_image(sample, save_file / 'generated_protein_img.png', normalize=True, value_range=(-1, 1))

        # msfs_gen.append(compute_msf_resolution((sample+1).mul(127.5).add_(0.5).clamp_(0, 255).squeeze().cpu().numpy().astype(np.uint8), pixel_size=206))
        # msfs_real.append(compute_msf_resolution((protein_img+1).mul(127.5).add_(0.5).clamp_(0, 255).squeeze().cpu().numpy().astype(np.uint8), pixel_size=206))

        iou = compute_iou(binarize_img(sample, threshold_mode="quantile", quantile_q=0.5), binarize_img(protein_img)).item()
        ious.append(iou)

        generated_threshold_img = binarize_img(sample, threshold_mode="quantile", quantile_q=0.5)
        real_threshold_img = binarize_img(protein_img)

        real_threshold_img = 2 * (real_threshold_img.float() * 0.5) - 1
        real_threshold_img = torch.cat([torch.full_like(protein_img, -1), real_threshold_img, nucleus_img], dim=1)
        save_image(real_threshold_img, save_file / 'real_threshold_img.png', normalize=True, value_range=(-1, 1))        

        generated_threshold_img = 2 * (generated_threshold_img.float() * 0.5) - 1
        generated_threshold_img = torch.cat([torch.full_like(protein_img, -1), generated_threshold_img, nucleus_img], dim=1)
        save_image(generated_threshold_img, save_file / 'generated_threshold_img.png', normalize=True, value_range=(-1, 1))

        save_colored_image((nucleus_img.squeeze() + 1) / 2, save_file / 'real_nucleus_img_blue.png', 'blue')
        save_colored_image((protein_img.squeeze() + 1) / 2, save_file / 'real_protein_img_green.png', 'green')
        save_colored_image((sample.squeeze() + 1) / 2, save_file / 'generated_protein_img_green.png', 'green')

        logger.info(f"iou: {iou}")

    logger.info(f"Avg IoU: {sum(ious)/len(ious)}")
    # logger.info(f"Avg MSF of Generated Image: {sum(msfs_gen)/len(msfs_gen)}")
    # logger.info(f"Avg MSF of Real Image: {sum(msfs_real)/len(msfs_real)}")

    msfs_gen = []
    gen_img_paths = sorted(glob.glob(f'./{output_dir}/*/generated_protein_img.png'))
    for img_path in gen_img_paths:
        img = np.array(Image.open(img_path).convert('L')).astype(np.float64) / 255
        msfs_gen.append(compute_msf_resolution(img, pixel_size=206))

    msfs_real = []
    real_img_paths = sorted(glob.glob(f'./{output_dir}/*/real_protein_img.png'))
    for img_path in real_img_paths:
        img = np.array(Image.open(img_path).convert('L')).astype(np.float64) / 255
        msfs_real.append(compute_msf_resolution(img, pixel_size=206))

    logger.info(f"Avg MSF of Generated Image: {sum(msfs_gen)/len(msfs_gen)} nm")
    logger.info(f"Avg MSF of Real Image: {sum(msfs_real)/len(msfs_real)} nm")

    real_img_files = sorted(glob.glob(f'./{output_dir}/*/real_img.png'))
    generated_img_files = sorted(glob.glob(f'./{output_dir}/*/generated_img.png'))

    new_real_img_path = output_dir / 'compute_fid/real_img'
    new_generated_img_path = output_dir / 'compute_fid/generated_img'

    put_imgs_to_new_file(new_real_img_path, real_img_files)
    put_imgs_to_new_file(new_generated_img_path, generated_img_files)

    real_img_files = sorted(glob.glob(f'./{output_dir}/*/real_threshold_img.png'))
    generated_img_files = sorted(glob.glob(f'./{output_dir}/*/generated_threshold_img.png'))
        
    new_real_img_path = output_dir / 'compute_fid/real_threshold_img'
    new_generated_img_path = output_dir / './compute_fid/generated_threshold_img'

    put_imgs_to_new_file(new_real_img_path, real_img_files)
    put_imgs_to_new_file(new_generated_img_path, generated_img_files)


if __name__ == "__main__":
    main()
