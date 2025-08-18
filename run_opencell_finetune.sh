# Using Weights & Biases (WandB) for experiment tracking (optional)
# export WANDB_RUN_NAME=       # (Optional) Name of the current run (visible in the WandB dashboard)
# export WANDB_API_KEY=        # (Optional) Your WandB API key to authenticate logging
# export WANDB_PROJECT=        # (Optional) WandB project name to group related experiments

# Set the output directory for training results
export output_dir=./finetune_opencell/ft_cell_diff_opencell

# Set the path to the training dataset
export data_path=./processed_datasets/opencell/lmdb_dataset

# Set the path to the pretrained VAE checkpoint
export vae_loadcheck_path=./pretrained_models/vae/opencell_finetuned.bin

# Set the path to the HPA pretrained CELL-Diff model
export loadcheck_path=./pretrained_models/cell_diff/hpa_pretrained.bin

bash scripts/cell_diff/finetune_opencell.sh