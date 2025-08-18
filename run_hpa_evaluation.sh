# Set the output directory for training results
export output_dir=./output/evaluate_hpa

# Set the path to the training dataset
export data_path=./processed_datasets/HPA/lmdb_dataset

# Set the path to the pretrained VAE checkpoint
export vae_loadcheck_path=./pretrained_models/vae/hpa_pretrained.bin

# Set the path to the HPA pretrained CELL-Diff model
export loadcheck_path=./pretrained_models/cell_diff/hpa_pretrained.bin

bash scripts/cell_diff/evaluate_hpa.sh