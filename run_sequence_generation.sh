# Set the path for cell morphology images
# Available options: data/opencell/nls (for nuclear localization signal generation), data/opencell/nes (for nuclear export signal generation).
export image_path=./data/opencell/nls

# Set the number of amino acids to generate.
export num_aas=15

# Set the path to the VAE checkpoint
export vae_loadcheck_path=./pretrained_models/vae/opencell_finetuned.bin

# Set the path to the model weights
export loadcheck_path=./pretrained_models/cell_diff/opencell_finetuned_all.bin

# Set the random seed
export seed=6

# Run the image generation script
bash scripts/cell_diff/evaluate_seq_generation.sh