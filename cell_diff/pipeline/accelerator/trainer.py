# -*- coding: utf-8 -*-
import copy
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from cell_diff.logging import logger, metric_logger
from cell_diff.pipeline.accelerator.accelerator import (
    GroupedBatchIter, 
    Accelerator, 
    DdpAccelerator, 
    SingleNodeAccelerator, 
)
from cell_diff.pipeline.accelerator.dataclasses import (
    TrainerConfig, 
    TrainerState, 
    TrainLogOutput, 
    TrainStrategy, 
    ValidLogOutput, 
)
from cell_diff.pipeline.accelerator.model import Model

def seed_everything(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class LossAccumulator(object):
    def __init__(self):
        self.sum = 0
        self.num_examples = 0

    def add(self, loss, num_examples):
        if loss is None:
            return

        if type(loss) == torch.Tensor:
            loss = loss.item()

        if type(num_examples) == torch.Tensor:
            num_examples = num_examples.item()

        if num_examples is None or num_examples <= 0:
            return

        if np.isnan(loss) or np.isinf(loss):
            return

        self.sum += loss * num_examples
        self.num_examples += num_examples

    def reset(self):
        self.sum = 0.0
        self.num_examples = 0

    @property
    def averge_loss(self):
        if self.num_examples == 0:
            return 0
        return self.sum / self.num_examples


class LogAccumulator(object):
    def __init__(self, world_size=1, allreduce_fn=None):
        self.sum = 0
        self.num_examples = 0
        self.extra_log = {}
        self.extra_log_num = {}
        self.start_time = time.time()
        self.allreduce_fn = allreduce_fn
        self.world_size = world_size

    def add(self, loss, num_examples, extra_log=None):
        if loss is None:
            return

        if type(loss) == torch.Tensor:
            loss = loss.item()

        if type(num_examples) == torch.Tensor:
            num_examples = num_examples.item()

        if num_examples is None or num_examples <= 0:
            return

        if np.isnan(loss) or np.isinf(loss):
            return

        self.sum += loss * num_examples
        self.num_examples += num_examples

        if extra_log is not None:
            for k, v in extra_log.items():
                if k not in self.extra_log and isinstance(v, (torch.Tensor, float)):
                    if isinstance(v, torch.Tensor):
                        self.extra_log[k] = v.item() * num_examples
                    else:
                        self.extra_log[k] = v * num_examples
                    self.extra_log_num[k] = 1 * num_examples
                elif k in self.extra_log and isinstance(v, (torch.Tensor, float)):
                    if isinstance(v, torch.Tensor):
                        self.extra_log[k] += v.item() * num_examples
                    else:
                        self.extra_log[k] += v * num_examples
                    self.extra_log_num[k] += 1 * num_examples

    def reset(self):
        self.sum = 0.0
        self.num_examples = 0
        self.start_time = time.time()
        for k, v in self.extra_log.items():
            self.extra_log[k] = 0.0
            self.extra_log_num[k] = 0

    @property
    def averge_loss(self):
        if self.num_examples == 0:
            return 0
        return self.sum / self.num_examples

    def _allreducelog(self, log_dict: dict = {}, log_num_dict: dict = {}):
        return self.allreduce_fn(log_dict, log_num_dict)

    @property
    def averge_log(self):
        self.extra_log["SamplePerSec"] = self.num_examples / (
            time.time() - self.start_time
        )
        self.extra_log_num["SamplePerSec"] = 1.0 / self.world_size
        if self.world_size == 1 or self.allreduce_fn is None:
            return {k: v / self.extra_log_num[k] for k, v in self.extra_log.items()}
        else:
            return self._allreducelog(self.extra_log, self.extra_log_num)


class Trainer(object):
    def __init__(
        self,
        args: TrainerConfig,
        model: Model,
        train_data: Dataset,
        valid_data: Optional[Dataset] = None,
        test_data: Optional[Dataset] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        loss_log_dict: Optional[dict] = {},
    ):
        super().__init__()
        self.args = args

        logger.info("Trainer args: {}", args)

        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.accelerator = self.build_accelerator(loss_log_dict=loss_log_dict)
        self.accelerator.set_up()

        self.accelerator.build_data_loader(train_data, valid_data)

        self.state = TrainerState(args=args)

        self.save_dir = Path(self.args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if self.args.finetune_from_checkpoint_dir is not None:
            self.finetune_from_checkpoint_dir = Path(
                self.args.finetune_from_checkpoint_dir
            )
        else:
            self.finetune_from_checkpoint_dir = None

        self.world_size = self.accelerator.world_size
        self.start_iteration = 0

    def save_checkpoint(self, name: str, state: Union[TrainerState, dict]):
        if isinstance(state, TrainerState):
            self.accelerator.save_checkpoint(name, asdict(state))
        else:
            self.accelerator.save_checkpoint(name, state)
        self._save_rng_and_iter_state(self.save_dir)

    def _load_checkpoint(self, path: Path, model_states_only: bool = False):
        checkpoint_list_path = path / "checkpoint_list.txt"

        checkpoint_last = None
        if model_states_only and self.args.finetune_from_checkpoint_id is not None:
            checkpoint_last = self.args.finetune_from_checkpoint_id
        elif checkpoint_list_path.exists():
            with open(checkpoint_list_path, "r") as f:
                checkpoint_list = f.read().splitlines()
            if len(checkpoint_list) > 0:
                checkpoint_last = checkpoint_list[-1]

        if checkpoint_last is not None:
            checkpoint_path = path / checkpoint_last
            if checkpoint_path.exists():
                if not model_states_only:
                    logger.info(f"Resume from checkpoint: {checkpoint_path}")
                else:
                    logger.info(f"Finetune from checkpoint: {checkpoint_path}")
                self.state = self.accelerator.load_checkpoint(
                    path,
                    checkpoint_last,
                    self.state,
                    model_states_only=model_states_only,
                )
            else:
                logger.warning(f"Checkpoint path {checkpoint_path} does not exist.")
        else:
            logger.warning(
                f"Non-empty checkpoint_list.txt or latest file is not present in {path}, or finetune_from_checkpoint_id is not provided. No checkpoint is loaded."
            )

    def resume(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._load_checkpoint(self.save_dir)
        self.start_iteration = self._load_rng_and_iter_state(self.save_dir)

    def finetune_from_checkpoint(self):
        if self.finetune_from_checkpoint_dir is not None:
            self._load_checkpoint(
                self.finetune_from_checkpoint_dir, model_states_only=True
            )
        else:
            logger.warning("No finetune_from_checkpoint_dir is provided.")

    def build_accelerator(self, loss_log_dict: Optional[dict] = {}) -> Accelerator:
        if self.args.strategy == TrainStrategy.Single:
            return SingleNodeAccelerator(
                self.args,
                self.model,
                self.optimizer,
                self.lr_scheduler,
                "cpu" if self.args.cpu else "cuda",
            )
        elif self.args.strategy == TrainStrategy.DDP:
            return DdpAccelerator(
                self.args,
                self.model,
                self.optimizer,
                self.lr_scheduler,
            )
        else:
            raise ValueError(f"Unknown strategy: {self.args.strategy}")

    def build_log_output(self, loss, extra_output=None) -> TrainLogOutput:
        try:
            lr = self.accelerator.lr_scheduler.get_last_lr()[0]
        except:
            lr = 0.0

        if type(loss) == torch.Tensor:
            loss = loss.item()

        return TrainLogOutput(
            loss=loss,
            grad_scale=self.accelerator.grad_scale,
            lr=lr,
            epoch=self.state.epoch,
            batch=self.state.batch,
            global_step=self.state.global_step,
            extra_output=extra_output,
        )

    def should_stop(self) -> bool:
        if (
            self.args.total_num_epochs is not None
            and self.args.total_num_epochs > 0
            and self.state.epoch >= self.args.total_num_epochs
        ):
            return True
        if (
            self.args.total_num_steps is not None
            and self.args.total_num_steps > 0
            and self.state.global_step >= self.args.total_num_steps
        ):
            return True
        return False

    def should_save_batch_checkpoint(self) -> bool:
        return (
            self.args.save_batch_interval > 0
            and (self.state.global_step + 1) % self.args.save_batch_interval == 0
        )

    def should_save_epoch_checkpoint(self) -> bool:
        return (
            self.args.save_epoch_interval > 0
            and (self.state.epoch + 1) % self.args.save_epoch_interval == 0
        )

    def should_log(self) -> bool:
        return (
            self.args.log_interval > 0
            and self.state.global_step % self.args.log_interval == 0
        )

    def should_do_batch_validate(self) -> bool:
        return (
            self.args.val_batch_interval > 0
            and self.state.global_step % self.args.val_batch_interval == 0
        )

    def should_do_epoch_validate(self) -> bool:
        return (
            self.args.val_epoch_interval > 0
            and (self.state.epoch + 1) % self.args.val_epoch_interval == 0
        )

    @property
    def train_data_loader(self) -> DataLoader:
        """
        Return the training data loader.
        """
        return GroupedBatchIter(
            self.accelerator.train_data_loader,
            self.args.gradient_accumulation_steps,
            drop_last=True,
        )

    @property
    def valid_data_loader(self) -> DataLoader:
        return self.accelerator.valid_data_loader

    def train(self):
        """
        Train the model on the training data loader.
        """
        logger.info("Start training")
        logger.info(self.model)

        assert self.train_data_loader is not None

        self.model.before_training()
        if self.args.ifresume:
            self.resume()
        elif self.args.finetune_from_checkpoint_dir is not None:
            self.finetune_from_checkpoint()

        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logger.info(
            "Total number of parameters: {:,}, trainable: {:,}",
            total_num,
            trainable_num,
        )

        while (
            self.state.epoch < self.args.total_num_epochs
            and self.state.global_step < self.args.total_num_steps
        ):
            self.accelerator.before_epoch(self.state.epoch)

            logger.info("Start Training for epoch: {}", self.state.epoch)

            loss_accumulator = LossAccumulator()
            interval_loss_accumulator = LogAccumulator(
                self.accelerator.world_size, self.accelerator._allreducelog
            )

            # skip first batches
            data_iterator = self.train_data_loader
            data_iterator = self.skip_first_batches(data_iterator, self.start_iteration)

            for grouped_batch_data in data_iterator:
                model_output = self.accelerator.train_step(grouped_batch_data)
                loss_accumulator.add(model_output.loss, model_output.num_examples)
                interval_loss_accumulator.add(
                    model_output.loss,
                    model_output.num_examples,
                    model_output.log_output,
                )

                # Log and save checkpoint
                self.state.batch += 1
                self.state.global_step += 1

                if self.should_do_batch_validate():
                    self.validate()

                if self.should_log():
                    log_output = self.build_log_output(
                        # model_output.loss, model_output.log_output
                        interval_loss_accumulator.averge_loss,
                        interval_loss_accumulator.averge_log,
                    )
                    interval_loss_accumulator.reset()
                    metric_logger.log(log_output, "train_inner")

                if self.should_save_batch_checkpoint():
                    checkpoint_name = (
                        f"checkpoint_E{self.state.epoch}_B{self.state.batch}.pt"
                    )
                    self.save_checkpoint(checkpoint_name, self.state)

            log_output = self.build_log_output(loss_accumulator.averge_loss)
            metric_logger.log(log_output, "train")

            if self.should_do_epoch_validate():
                self.validate()

            self.accelerator.barrier()
            if self.should_save_epoch_checkpoint():
                checkpoint_name = f"checkpoint_E{self.state.epoch}.pt"
                self.save_checkpoint(checkpoint_name, self.state)

            self.state.epoch += 1
            self.state.batch = 0

        self.model.after_training()

        logger.info("Finished Training")

    def validate(self):
        """
        Validate the model on the validation data loader.
        """
        if self.valid_data_loader is None:
            logger.warning("No validation data, skip validation")
            return

        logger.info(
            "Start validation for epoch: {}, global step: {}",
            self.state.epoch,
            self.state.global_step,
        )

        loss_accumulator = LossAccumulator()
        interval_loss_accumulator = LogAccumulator(
            self.accelerator.world_size, self.accelerator._allreducelog
        )

        for idx, batch_data in enumerate(self.valid_data_loader):
            output = self.accelerator.valid_step(batch_data)
            loss_accumulator.add(output.valid_loss, output.num_examples)
            interval_loss_accumulator.add(
                output.valid_loss,
                output.num_examples,
                output.extra_output,
            )

            if (idx + 1) % self.args.val_batch_log_interval == 0:
                logger.info(
                    "Validtion batch: {} / {}, loss: {}",
                    idx + 1,
                    len(self.valid_data_loader),
                    output.valid_loss,
                )

        # DDP need to sync loss and num_examples at validation
        total_loss, num_examples = self.accelerator.sync_valid_loss(
            loss_accumulator.sum, loss_accumulator.num_examples
        )

        if num_examples > 0:
            valid_loss = total_loss / num_examples
        else:
            valid_loss = 0

        valid_log = ValidLogOutput(
            valid_loss=valid_loss,
            num_examples=num_examples,
            extra_output=interval_loss_accumulator.averge_log,
        )

        metric_logger.log(valid_log, "valid")

    def _save_rng_and_iter_state(self, checkpoint):
        """
        Save the RNG and iteration states to the checkpoint to resume training from break point.
        Args:
            checkpoint (str): the path to the checkpoint
        """
        if checkpoint is None:
            return

        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
            "cuda": torch.cuda.random.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
            "iteration": self.state.batch,
            "epoch": self.state.epoch,
        }

        if self.accelerator.world_size > 1:
            process_index = self.args.local_rank
            rng_file = os.path.join(checkpoint, f"rng_state_{process_index}.pth")
        else:
            rng_file = os.path.join(checkpoint, "rng_state.pth")

        torch.save(rng_state, rng_file)

    def _load_rng_and_iter_state(self, checkpoint):
        """
        Load the RNG and iteration states from the checkpoint to resume training from break point.
        Args:
            checkpoint (str): the path to the checkpoint
        """
        if checkpoint is None:
            return

        if self.accelerator.world_size > 1:
            process_index = self.args.local_rank
            rng_file = os.path.join(checkpoint, f"rng_state_{process_index}.pth")
            if not os.path.isfile(rng_file):
                logger.warning(
                    f"Didn't find an RNG file for process {process_index}, if you are resuming a training that "
                    "wasn't launched in a distributed fashion, reproducibility is not guaranteed."
                )
                return
        else:
            rng_file = os.path.join(checkpoint, "rng_state.pth")
            if not os.path.isfile(rng_file):
                logger.warning(
                    "Didn't find an RNG file, if you are resuming a training that was launched in a distributed "
                    "fashion, reproducibility is not guaranteed."
                )
                return

        checkpoint_rng_state = torch.load(rng_file)
        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])
        torch.random.set_rng_state(checkpoint_rng_state["cpu"])
        if torch.cuda.is_available():
            if self.accelerator.world_size > 1:
                torch.cuda.random.set_rng_state_all(checkpoint_rng_state["cuda"])
            else:
                try:
                    torch.cuda.random.set_rng_state(checkpoint_rng_state["cuda"])
                except Exception as e:
                    logger.warning(
                        f"Didn't manage to set back the RNG states of the GPU because of the following error:\n {e}"
                        "\nThis won't yield the same results as if the training had not been interrupted."
                    )

        if "epoch" in checkpoint_rng_state:
            self.state.epoch = checkpoint_rng_state["epoch"]

        start_iteration = checkpoint_rng_state["iteration"]

        return start_iteration

    def skip_first_batches(self, data_iterator, start_iteration=None):
        """
        Skip the first start_iteration batches in the training data loader to resume training from break point.
        Args:
            start_iteration (int): the number of batches to skip
        """
        if start_iteration is None or start_iteration == 0:
            return data_iterator

        logger.info(f"Skipping the first {start_iteration} batches")
        for i, _ in enumerate(data_iterator):
            if i == start_iteration:
                break

        self.start_iteration = 0
        return data_iterator