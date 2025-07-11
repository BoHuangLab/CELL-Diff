# Using Weights & Biases (WandB) for experiment tracking (optional)
# export WANDB_RUN_NAME=       # (Optional) Name of the current run (visible in the WandB dashboard)
# export WANDB_API_KEY=        # (Optional) Your WandB API key to authenticate logging
# export WANDB_PROJECT=        # (Optional) WandB project name to group related experiments

# Set the output directory for training results
export output_dir=pretrain_hpa/pt_cell_diff_hpa

# Set the path to the training dataset
export data_path=Path/to/HPA/lmdb_dataset

# Set the path to the pretrained VAE checkpoint
export vae_loadcheck_path=Path/to/pretrained_models/vae/hpa_pretrained.bin

bash scripts/cell_diff/pretrain_hpa.sh