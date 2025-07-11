# Set the output directory for training results
export output_dir=output/evaluate_opencell

# Set the path to the training dataset
export data_path=Path/to/opencell/lmdb_dataset

# Set the path to the OpenCell finetuned VAE checkpoint
export vae_loadcheck_path=Path/to/pretrained_models/vae/opencell_finetuned.bin

# Set the path to the OpenCell finetuned CELL-Diff model
export loadcheck_path=Path/to/pretrained_models/cell_diff/opencell_finetuned.bin

bash scripts/cell_diff/evaluate_opencell.sh