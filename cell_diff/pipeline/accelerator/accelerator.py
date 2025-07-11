# -*- coding: utf-8 -*-
import multiprocessing
import os
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Union

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from cell_diff.logging import logger
from cell_diff.pipeline.accelerator.dataclasses import (
    ModelOutput,
    TrainerState,
    ValidLogOutput,
)
from cell_diff.pipeline.accelerator.fp16_scaler import FP16Scaler
from cell_diff.utils.move_to_device import move_to_device


class GroupedBatchIter(object):
    """
    This class is used to group batches into a larger batch. i.e., gradient accumulation.
    """

    def __init__(self, it, group_size, drop_last=False):
        self.it = it
        self.group_size = group_size
        self.drop_last = drop_last

    def __iter__(self):
        chunk = []
        for item in self.it:
            chunk.append(item)
            if len(chunk) == self.group_size:
                yield chunk
                chunk = []
        if not self.drop_last and chunk:
            yield chunk

    def __len__(self):
        if self.drop_last:
            return len(self.it) // self.group_size
        else:
            return (len(self.it) + self.group_size - 1) // self.group_size

class Accelerator(ABC):
    @abstractmethod
    def set_up():
        pass

    @abstractmethod
    def train_step(self, grouped_batch_data):
        pass

    @abstractmethod
    def valid_step(self, batch_data):
        pass

    @abstractmethod
    def save_checkpoint(self, ckpt_id, extra_state):
        pass

    @abstractmethod
    def load_checkpoint(
        self,
        ckpt_dir,
        ckpt_id,
        trainer_state,
        model_states_only: bool = False,
    ) -> TrainerState:
        pass

    @abstractmethod
    def build_data_loader(self, train_data, val_data):
        pass

    @abstractmethod
    def barrier(self):
        pass

    @abstractmethod
    def sync_valid_loss(self, total_loss, num_examples):
        pass

    @property
    @abstractmethod
    def grad_scale(self) -> float:
        pass

    def before_epoch(self, epoch: int):
        pass


class SingleNodeAccelerator(Accelerator):
    def __init__(self, args, model, optimizer, lr_scheduler, device: str) -> None:
        super().__init__()
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.world_size = 1

        if not torch.cuda.is_available():
            self.device = "cpu"
        self.scaler = FP16Scaler(
            init_scale=self.args.grad_scaler_init, enabled=self.args.fp16
        )

        if args.fp16:
            self.model = self.model.half()

    @property
    def grad_scale(self) -> float:
        return self.scaler.scale

    def set_up(self):
        if self.optimizer is None:
            self.optimizer, self.lr_scheduler = self.model.config_optimizer()

    def barrier(self):
        pass

    def build_data_loader(self, train_data, valid_data=None):
        self.train_sampler = RandomSampler(train_data)
        self.train_data_loader = DataLoader(
            train_data,
            sampler=self.train_sampler,
            batch_size=self.args.train_batch_size,
            collate_fn=train_data.collate,
            drop_last=True,
        )

        if valid_data:
            self.valid_data_loader = DataLoader(
                valid_data,
                sampler=None,
                batch_size=self.args.val_batch_size,
                collate_fn=valid_data.collate,
                drop_last=False,
            )
        else:
            self.valid_data_loader = None

    def train_step(self, grouped_batch_data) -> ModelOutput:
        assert grouped_batch_data, "grouped_batch_data is empty"

        self.model.train()
        self.model.to(self.device)

        self.optimizer.zero_grad()
        success_batch_count = 0
        for batch_data in grouped_batch_data:
            self.model.before_batch()
            batch_data = move_to_device(batch_data, self.device)

            pred = self.model(batch_data)
            model_output = self.model.compute_loss(pred, batch_data)
            loss = model_output.loss / len(grouped_batch_data)

            if torch.isnan(loss).item() or torch.isinf(loss).item():
                logger.info("loss is nan or inf. skip this batch")
                continue
            else:
                success_batch_count += 1
                self.scaler.backward(loss)

            self.model.after_batch()

        if success_batch_count > 0:
            self.scaler.step(self.model, self.optimizer, self.args.gradient_clipping)

        self.lr_scheduler.step()

        return model_output

    def valid_step(self, batch_data) -> ValidLogOutput:
        self.model.eval()
        self.model.to(self.device)

        batch_data = move_to_device(batch_data, self.device)
        with torch.no_grad():
            pred = self.model(batch_data)
            model_output = self.model.compute_loss(pred, batch_data)

        if hasattr(batch_data, "batch_size"):
            num_examples = batch_data.batch_size
        elif hasattr(model_output, "num_examples"):
            num_examples = model_output.num_examples
        else:
            logger.info("num_examples is not found. set to None")
            num_examples = None

        return ValidLogOutput(
            valid_loss=model_output.loss.item(),
            num_examples=num_examples,
            extra_output=model_output.log_output,
        )

    def save_checkpoint(self, ckpt_id: str, extra_state: Optional[dict] = None):
        save_dir = Path(self.args.save_dir)

        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }

        if extra_state is not None:
            checkpoint.update(extra_state)
        logger.info("save checkpoint: {}", ckpt_id)
        torch.save(checkpoint, save_dir / ckpt_id)

        with open(save_dir / "checkpoint_list.txt", "a") as f:
            f.write(ckpt_id + "\n")

    def load_checkpoint(
        self,
        ckpt_dir: Path,
        ckpt_id: Union[int, str],
        trainer_state: TrainerState,
        model_states_only: bool = False,
    ) -> TrainerState:
        checkpoint_path = ckpt_dir / str(ckpt_id)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])
        if not model_states_only:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        if not model_states_only:
            for k, v in checkpoint.items():
                if k not in ["model", "optimizer", "lr_scheduler"]:
                    setattr(trainer_state, k, v)
        return trainer_state

    def sync_valid_loss(self, total_loss, num_examples):
        return total_loss, num_examples

    @staticmethod
    def _allreducelog(log_dict: dict = {}, log_num_dict: dict = {}):
        return None


class DdpAccelerator(SingleNodeAccelerator):
    def __init__(self, args, model, optimizer, lr_scheduler) -> None:
        super().__init__(args, model, optimizer, lr_scheduler, device="cuda")

    def set_up(self):
        super().set_up()
        assert "WORLD_SIZE" in os.environ, "WORLD_SIZE must be set to use DDP"
        assert "RANK" in os.environ, "RANK must be set to use DDP"
        assert "LOCAL_RANK" in os.environ, "LOCAL_RANK must be set to use DDP"

        self.world_size = int(os.environ["WORLD_SIZE"])
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])

        master_addr = os.environ.get("MASTER_ADDR", "")
        master_port = os.environ.get("MASTER_PORT", "")

        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)

        multiprocessing.set_start_method("spawn", force=True)

        logger.critical(
            f"Initializing DDP by env://. word size: {self.world_size}, rank: {self.rank}, local_rank: {self.local_rank}, master_addr: {master_addr}, master_port: {master_port}"
        )
        torch.distributed.init_process_group(
            backend=self.args.dist_backend,
            init_method="env://",
            world_size=self.world_size,
            rank=self.rank,
        )
        torch.distributed.barrier()
        
        logger.success("DDP initialized.")

        self.model.to(self.device)
        self.ddp_model = DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=True,
        )

    def barrier(self):
        torch.distributed.barrier()

    def train_step(self, grouped_batch_data) -> ModelOutput:
        assert grouped_batch_data, "grouped_batch_data is empty"

        self.ddp_model.train()
        self.optimizer.zero_grad()

        success_batch_count = 0
        for idx, batch_data in enumerate(grouped_batch_data):
            self.model.before_batch()
            batch_data = move_to_device(batch_data, self.device)

            # No sync for gradient accumulation
            maybe_no_sync = (
                self.ddp_model.no_sync()
                if idx != len(grouped_batch_data) - 1
                else nullcontext()
            )

            with maybe_no_sync:
                pred = self.ddp_model(batch_data)
                model_output = self.model.compute_loss(pred, batch_data)
                loss = model_output.loss / len(grouped_batch_data)

                if torch.isnan(loss).item() or torch.isinf(loss).item():
                    logger.info("loss is nan or inf. skip this batch")
                    continue
                else:
                    success_batch_count += 1
                    self.scaler.backward(loss)

            self.model.after_batch()

        if success_batch_count > 0:
            self.scaler.step(self.model, self.optimizer, self.args.gradient_clipping)
        
        self.lr_scheduler.step()

        return model_output

    def build_data_loader(self, train_data, val_data):
        
        train_batch_size_per_gpu = self.args.train_batch_size // (
            self.world_size * self.args.gradient_accumulation_steps
        )
        assert (
            train_batch_size_per_gpu > 0
        ), "train_batch_size_per_gpu should be greater than 0"

        self.train_sampler = DistributedSampler(
            train_data, num_replicas=self.world_size, rank=self.rank
        )
        self.train_data_loader = DataLoader(
            train_data,
            sampler=self.train_sampler,
            batch_size=train_batch_size_per_gpu,
            collate_fn=train_data.collate,
            drop_last=True,
            num_workers=self.args.num_workers
        )

        if val_data:
            valid_batch_size_per_gpu = self.args.val_batch_size // (
                self.world_size * self.args.gradient_accumulation_steps
            )
            assert (
                valid_batch_size_per_gpu > 0
            ), "train_batch_size_per_gpu should be greater than 0"

            validsampler = torch.utils.data.distributed.DistributedSampler(
                val_data, num_replicas=self.world_size, shuffle=False
            )
            self.valid_data_loader = DataLoader(
                val_data,
                sampler=validsampler,
                batch_size=valid_batch_size_per_gpu,
                collate_fn=val_data.collate,
                drop_last=False,
                num_workers=self.args.num_workers
            )
        else:
            self.valid_data_loader = None

    def before_epoch(self, epoch: int):
        self.train_sampler.set_epoch(epoch)

    def save_checkpoint(self, ckpt_id: str, extra_state: Optional[dict] = None):
        if self.rank == 0:
            super().save_checkpoint(ckpt_id, extra_state)

        torch.distributed.barrier()

    def sync_valid_loss(self, total_loss, num_examples):
        total_loss = torch.Tensor([total_loss]).cuda(self.device)
        num_examples = torch.Tensor([num_examples * 1.0]).cuda(self.device)
        torch.distributed.all_reduce(total_loss)
        torch.distributed.all_reduce(num_examples)
        total_loss = total_loss.item()
        num_examples = num_examples.item()

        return total_loss, num_examples

    @staticmethod
    def _allreducelog(log_dict: dict = {}, log_num_dict: dict = {}):
        for k, v in log_dict.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            v = v.cuda()
            torch.distributed.all_reduce(v, op=torch.distributed.ReduceOp.SUM)
            log_dict[k] = v.item()

        for k, v in log_num_dict.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            v = v.cuda()
            torch.distributed.all_reduce(v, op=torch.distributed.ReduceOp.SUM)
            log_num_dict[k] = v.item()

        return {k: v / log_num_dict[k] for k, v in log_dict.items()}

