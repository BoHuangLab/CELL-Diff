# Set the output directory for training results
export output_dir=output/evaluate_hpa

# Set the path to the training dataset
export data_path=Path/to/HPA/lmdb_dataset

# Set the path to the pretrained VAE checkpoint
export vae_loadcheck_path=Path/to/pretrained_models/vae/hpa_pretrained.bin

# Set the path to the HPA pretrained CELL-Diff model
export loadcheck_path=Path/to/pretrained_models/cell_diff/hpa_pretrained.bin

bash scripts/cell_diff/evaluate_hpa.sh