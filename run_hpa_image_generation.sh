# Set the output directory
export output_dir=./output/

# Set the path for cell morphology images
# Available options: './data/hpa/1/', './data/hpa/2/', './data/hpa/3/', './data/hpa/4/', './data/hpa/5/'
export image_path=./data/hpa/1/

# Specify the target protein sequence (TUBB4B)
export test_sequence=MREIVHLQAGQCGNQIGAKFWEVISDEHGIDPTGTYHGDSDLQLERINVYYNEATGGNYVPRAVLVDLEPGTMDSVRSGPFGQIFRPDNFVFGQSGAGNNWAKGHYTEGAELVDAVLDVVRKEAESCDCLQGFQLTHSLGGGTGSGMGTLLISKIREEFPDRIMNTFSVVPSPKVSDTVVEPYNATLSVHQLVENTDETYCIDNEALYDICFRTLKLTTPTYGDLNHLVSATMSGVTTCLRFPGQLNADLRKLAVNMVPFPRLHFFMPGFAPLTSRGSQQYRALTVPELTQQMFDAKNMMAACDPRHGRYLTVAAVFRGRMSMKEVDEQMLSVQSKNSSYFVEWIPNNVKTAVCDIPPRGLKMAATFIGNSTAIQELFKRISEQFTAMFRRKAFLHWYTGEGMDEMEFTEAESNMNDLVSEYQQYQDATAEEGEFEEEAEEEVA

# Set the path to the pretrained VAE checkpoint
export vae_loadcheck_path=./pretrained_models/vae/hpa_pretrained.bin

# Specify the path to the pretrained model weights
export loadcheck_path=./pretrained_models/cell_diff/hpa_pretrained_all.bin

# Set the random seed
export seed=6

# Run the image generation script
bash scripts/cell_diff/evaluate_img_generation_hpa.sh