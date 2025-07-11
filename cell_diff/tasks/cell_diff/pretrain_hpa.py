# -*- coding: utf-8 -*-
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from cell_diff.criterions.unidiff import UniDiffCriterions
from cell_diff.data.hpa_data.dataset import HPALMDBDataset
from cell_diff.models.cell_diff.config import CELLDiffConfig
from cell_diff.models.cell_diff.model import CELLDiffModel
from cell_diff.models.vae.vae_model import VAEModel
from cell_diff.models.vae.vae_config import VAEConfig
from cell_diff.utils.cli_utils import cli
from transformers import Trainer, TrainingArguments
from cell_diff.logging.loggers import CELLDiffLoggingCallback
from cell_diff.logging import logger
from copy import deepcopy


@cli(CELLDiffConfig)
def main(args) -> None:

    vae_args = deepcopy(args)
    vae_args.infer = True

    vae = VAEModel(config=VAEConfig(**vars(vae_args)))

    for param in vae.parameters():
        param.requires_grad = False
    vae.eval()

    trainset = HPALMDBDataset(args, split_key=args.split_key, vae=vae)
    model = CELLDiffModel(config=CELLDiffConfig(**vars(args)), loss_fn=UniDiffCriterions)

    logger.info(args)

    training_args = TrainingArguments(
        output_dir=args.output_dir, 
        learning_rate=args.learning_rate, 
        weight_decay=args.weight_decay, 
        gradient_accumulation_steps=args.gradient_accumulation_steps, 
        per_device_train_batch_size=args.per_device_train_batch_size, 
        per_device_eval_batch_size=args.per_device_eval_batch_size, 
        num_train_epochs=args.num_train_epochs, 
        fp16=args.fp16, 
        bf16=args.bf16, 
        logging_dir=args.logging_dir, 
        logging_steps=args.logging_steps, 
        max_steps=args.max_steps, 
        warmup_steps=args.warmup_steps, 
        save_steps=args.save_steps, 
        seed=args.seed, 
        dataloader_num_workers=args.dataloader_num_workers, 
        report_to='wandb', 
        disable_tqdm=True, 
        remove_unused_columns=False, 
        overwrite_output_dir=True, 
        log_level='debug', 
        include_inputs_for_metrics=False, 
        save_safetensors=False, 
    )

    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=trainset, 
        data_collator=trainset.collate, 
        callbacks=[CELLDiffLoggingCallback()], 
    )
        
    trainer.train(resume_from_checkpoint=args.ifresume)

if __name__ == "__main__":
    main()

