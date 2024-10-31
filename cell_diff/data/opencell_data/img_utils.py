import torchvision.transforms.functional as TF
import random
from typing import Sequence
import torch

class RandomRotation:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


def replace_outliers(image, percentile=0.0001):
    lower_bound, upper_bound = torch.quantile(image, percentile), torch.quantile(
        image, 1 - percentile
    )
    mask = (image <= upper_bound) & (image >= lower_bound)
    valid_pixels = image[mask]
    image[~mask] = torch.clip(image[~mask], min(valid_pixels), max(valid_pixels))

    return image