# -*- coding: utf-8 -*-
import os
import sys
import shutil
import glob

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from pathlib import Path

from cell_diff.criterions.unidiff import UniDiffCriterions
from cell_diff.data.hpa_data.dataset import HPALMDBDataset
from cell_diff.models.cell_diff.cell_diff_config import CELLDiffConfig
from cell_diff.models.cell_diff.cell_diff_model import CELLDiffModel
from cell_diff.pipeline.accelerator.dataclasses import DistributedTrainConfig
from cell_diff.utils.cli_utils import cli
from cell_diff.logging import logger

from cell_diff.metrics.iou import compute_iou, binarize_img

from torchvision.utils import save_image
import numpy as np
from frc.deps_types import dip


def put_imgs_to_new_file(new_file_path, img_files):
    
    os.makedirs(new_file_path, exist_ok=True)
    
    for img_file in img_files:
        new_name = img_file.split('/')[-2] + '_' + img_file.split('/')[-1]
        
        new_file = os.path.join(new_file_path, new_name)        
        shutil.copy(img_file, new_file)

def compute_2d_psd_fft(image):
    """
    Compute the 2D Power Spectral Density (PSD) using FFT.

    Args:
        image (2D numpy array): Input 2D signal (e.g., an image)

    Returns:
        psd_2d (2D numpy array): 2D power spectral density
        freq_x (1D numpy array): Frequency axis for the x-direction
        freq_y (1D numpy array): Frequency axis for the y-direction
    """
    # Get the dimensions of the image
    ny, nx = image.shape
    
    # 2D FFT and shift the zero frequency component to the center
    fft_2d = np.fft.fft2(image)
    fft_2d_shifted = np.fft.fftshift(fft_2d)
    
    # Compute the 2D Power Spectral Density
    psd_2d = np.abs(fft_2d_shifted) ** 2 / (nx * ny)
    
    # Frequency ranges
    freq_x = np.fft.fftshift(np.fft.fftfreq(nx))
    freq_y = np.fft.fftshift(np.fft.fftfreq(ny))
    
    return psd_2d, freq_x, freq_y

def compute_msf_resolution(img, total_intensity=1000, image_size=256, threshold=1e-3, pixel_size=320):
    img = total_intensity * img / img.sum()

    psd_2d = compute_2d_psd_fft(img)[0]
    psd_1d = dip.RadialSum(psd_2d, None)
    psd_1d = np.array(psd_1d)

    psd_1d = psd_1d[:image_size // 2]

    index = next((i for i, val in enumerate(psd_1d) if val < threshold), None)

    if index is None:
        index = image_size // 2

    frequency = index / (image_size * pixel_size) 
    resolution = 1 / frequency

    return resolution

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

@cli(DistributedTrainConfig, CELLDiffConfig)
def main(args) -> None:
    if not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda"

    valset = HPALMDBDataset(args, split_key=args.split_key)
    vocab = valset.vocab
    
    if args.model_type == 'simple_diffusion':
        args.num_res_block = [int(a) for a in args.num_res_block.split(',')]
        args.dims = [int(a) for a in args.dims.split(',')]
        model = CELLDiffModel(args, loss_fn=UniDiffCriterions)

    model.to(device)
    model.eval()

    save_dir = Path(args.save_dir)
    save_dir = save_dir / args.split_key

    ious = []
    msfs_real = []
    msfs_gen = []

    for i, data in enumerate(valset):

        protein_seq = data['protein_seq'].unsqueeze(0).to(device)

        if protein_seq.shape[1] > args.max_protein_sequence_len:
            continue

        protein_img = data['protein_img'].unsqueeze(0).to(device)
        nucleus_img = data['nucleus_img'].unsqueeze(0).to(device)
        microtubules_img = data['microtubules_img'].unsqueeze(0).to(device)
        ER_img = data['ER_img'].unsqueeze(0).to(device)

        if args.cell_image == 'nucl':
            cell_img = nucleus_img
        elif args.cell_image == 'nucl,er':
            if args.test_cell_image == 'nucl':
                ER_img = torch.full_like(nucleus_img, -1)
            cell_img = torch.cat([nucleus_img, ER_img], dim=1)
        elif args.cell_image == 'nucl,mt':
            if args.test_cell_image == 'nucl':
                microtubules_img = torch.full_like(nucleus_img, -1)
            cell_img = torch.cat([nucleus_img, microtubules_img], dim=1)
        elif args.cell_image == 'nucl,er,mt':
            if args.test_cell_image == 'nucl':
                ER_img = torch.full_like(nucleus_img, -1)
                microtubules_img = torch.full_like(nucleus_img, -1)
            elif args.test_cell_image == 'nucl,er':
                microtubules_img = torch.full_like(nucleus_img, -1)
            elif args.test_cell_image == 'nucl,mt':
                ER_img = torch.full_like(nucleus_img, -1)
            cell_img = torch.cat([nucleus_img, ER_img, microtubules_img], dim=1)
        else:
            raise ValueError(f"Cell image type: {args.cell_image} is not supported")
        
        logger.info(data['ensg_id'])

        save_file = save_dir / ('{:04d}_'.format(i+1) + data['ensg_id'])
        save_file.mkdir(parents=True, exist_ok=True)
        
        real_protein_seq = vocab.untokenize(protein_seq.squeeze()[1:-1])
        logger.info(real_protein_seq)
        
        real_img = torch.cat([torch.full_like(protein_img, -1), protein_img, nucleus_img], dim=1)
        save_image(real_img, save_file / 'real_img.png', normalize=True, value_range=(-1, 1))
        save_image(nucleus_img, save_file / 'real_nucleus_img.png', normalize=True, value_range=(-1, 1))
        save_image(protein_img, save_file / 'real_protein_img.png', normalize=True, value_range=(-1, 1))
        
        sample = model.sequence_to_image(protein_seq, cell_img, sampling_strategy="ddim")
        pred_img = torch.cat([torch.full_like(protein_img, -1), sample, nucleus_img], dim=1)

        save_image(pred_img, save_file / 'generated_img.png', normalize=True, value_range=(-1, 1))
        save_image(sample, save_file / 'generated_protein_img.png', normalize=True, value_range=(-1, 1))

        msfs_gen.append(compute_msf_resolution(((sample + 1) / 2).squeeze().cpu().numpy()))
        msfs_real.append(compute_msf_resolution(((protein_img + 1) / 2).squeeze().cpu().numpy()))

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
        save_colored_image((microtubules_img.squeeze() + 1) / 2, save_file / 'real_microtubules_img_red.png', 'red')
        save_colored_image((protein_img.squeeze() + 1) / 2, save_file / 'real_protein_img_green.png', 'green')
        save_colored_image((ER_img.squeeze() + 1) / 2, save_file / 'real_ER_img_yellow.png', 'yellow')
        save_colored_image((sample.squeeze() + 1) / 2, save_file / 'generated_protein_img_green.png', 'green')

        logger.info(f"iou: {iou}")
    
    logger.info(f"Avg IoU: {sum(ious)/len(ious)}")
    logger.info(f"Avg MSF of Generated Image: {sum(msfs_gen)/len(msfs_gen)}")
    logger.info(f"Avg MSF of Real Image: {sum(msfs_real)/len(msfs_real)}")

    real_img_files = sorted(glob.glob(f'./{save_dir}/*/real_img.png'))
    generated_img_files = sorted(glob.glob(f'./{save_dir}/*/generated_img.png'))

    new_real_img_path = save_dir / 'compute_fid/real_img'
    new_generated_img_path = save_dir / 'compute_fid/generated_img'

    put_imgs_to_new_file(new_real_img_path, real_img_files)
    put_imgs_to_new_file(new_generated_img_path, generated_img_files)

    real_img_files = sorted(glob.glob(f'./{save_dir}/*/real_threshold_img.png'))
    generated_img_files = sorted(glob.glob(f'./{save_dir}/*/generated_threshold_img.png'))
        
    new_real_img_path = save_dir / 'compute_fid/real_threshold_img'
    new_generated_img_path = save_dir / './compute_fid/generated_threshold_img'

    put_imgs_to_new_file(new_real_img_path, real_img_files)
    put_imgs_to_new_file(new_generated_img_path, generated_img_files)

if __name__ == "__main__":
    main()
