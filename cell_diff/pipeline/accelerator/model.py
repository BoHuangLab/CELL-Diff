# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Optional, Tuple

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from cell_diff.pipeline.accelerator.dataclasses import ModelOutput


class Model(nn.Module, ABC):
    @abstractmethod
    def compute_loss(self, pred, batch) -> ModelOutput:
        pass

    @abstractmethod
    def config_optimizer(
        self, model: Optional[nn.Module]
    ) -> Tuple[Optimizer, LRScheduler]:
        """
        Return the optimizer and learning rate scheduler for this model.

        Returns:
            Tuple[Optimizer, LRScheduler]:
        """
        pass

    def before_training(self):
        """
        This method is called before training so you can do some initialization.
        For example, freeze some layers or set some layers to eval mode.
        """

        pass

    def after_training(self):
        """
        This method is called after training so you can do some finalization.
        """

        pass

    def before_batch(self):
        """
        This method is called before each batch so you can do some preprocessing.
        For example, set some layers to eval mode to disable dropout.
        """

        pass

    def after_batch(self):
        """
        This method is called after each batch so you can do some postprocessing.
        For example, set some layers to train mode to enable dropout.
        """

        pass
