# Set the output directory for training results
export output_dir=./output/evaluate_opencell

# Set the path to the training dataset
export data_path=./processed_datasets/opencell/lmdb_dataset

# Set the path to the OpenCell finetuned VAE checkpoint
export vae_loadcheck_path=./pretrained_models/vae/opencell_finetuned.bin

# Set the path to the OpenCell finetuned CELL-Diff model
export loadcheck_path=./pretrained_models/cell_diff/opencell_finetuned.bin

bash scripts/cell_diff/evaluate_opencell.sh