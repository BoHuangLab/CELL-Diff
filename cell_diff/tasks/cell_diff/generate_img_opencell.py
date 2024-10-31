# -*- coding: utf-8 -*-
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from pathlib import Path

from cell_diff.criterions.unidiff import UniDiffCriterions
from cell_diff.models.cell_diff.cell_diff_config import CELLDiffConfig
from cell_diff.models.cell_diff.cell_diff_model import CELLDiffModel
from cell_diff.tasks.cell_diff.test_config import TestConfig
from cell_diff.pipeline.accelerator.dataclasses import DistributedTrainConfig
from cell_diff.utils.cli_utils import cli_eval
from torchvision.utils import save_image

from PIL import Image
from torchvision.transforms.functional import to_tensor
from cell_diff.data.hpa_data.vocabulary import Alphabet, convert_string_sequence_to_int_index

def prepare_img_opencell(dataroot: Path):
    nucleus_img = to_tensor(Image.open(os.path.join(dataroot / "nucleus.png")).convert("L"))

    cell_img = nucleus_img.unsqueeze(0)
    cell_img = cell_img * 2 - 1

    return cell_img

@cli_eval(DistributedTrainConfig, CELLDiffConfig, TestConfig)
def main(args) -> None:
    if not torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cuda"
    
    args.num_res_block = [int(a) for a in args.num_res_block.split(',')]
    args.dims = [int(a) for a in args.dims.split(',')]
    model = CELLDiffModel(args, loss_fn=UniDiffCriterions)

    model.to(device)
    model.eval()
    
    cell_img = prepare_img_opencell(Path(args.cell_morphology_image_path)).to(device)

    test_sequence = args.test_sequence

    vocab = Alphabet()
    test_sequence_token = convert_string_sequence_to_int_index(vocab, test_sequence)
    test_sequence_token = torch.LongTensor(test_sequence_token).unsqueeze(0).to(device)

    print("Generating protein image from sequence: ")
    print(test_sequence)

    pred_protein_img = model.sequence_to_image(test_sequence_token, cell_img, sampling_strategy="ddim")

    save_dir = Path(args.save_dir)

    save_dir.mkdir(exist_ok=True)
    save_image(pred_protein_img, save_dir / 'pred_protein_img.png', normalize=True, value_range=(-1, 1))

    cat_img = torch.cat([torch.full_like(pred_protein_img, -1), pred_protein_img, cell_img], dim=1)
    save_image(cat_img, save_dir / 'pred_protein_img_cat.png', normalize=True, value_range=(-1, 1))

if __name__ == "__main__":
    main()