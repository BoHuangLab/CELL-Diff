# -*- coding: utf-8 -*-
from dataclasses import dataclass

@dataclass
class TestConfig:
    cell_morphology_image_path: str = '.'
    test_sequence: str = '.'