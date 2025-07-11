# Set the output directory
export output_dir=output/

# Set the path for cell morphology images
# Available options: './data/opencell/1/', './data/opencell/2/', './data/opencell/3/', './data/opencell/4/', './data/opencell/5/'
export image_path=data/opencell/1/

# Specify the target protein sequence (TUBB4B)
export test_sequence=MREIVHLQAGQCGNQIGAKFWEVISDEHGIDPTGTYHGDSDLQLERINVYYNEATGGNYVPRAVLVDLEPGTMDSVRSGPFGQIFRPDNFVFGQSGAGNNWAKGHYTEGAELVDAVLDVVRKEAESCDCLQGFQLTHSLGGGTGSGMGTLLISKIREEFPDRIMNTFSVVPSPKVSDTVVEPYNATLSVHQLVENTDETYCIDNEALYDICFRTLKLTTPTYGDLNHLVSATMSGVTTCLRFPGQLNADLRKLAVNMVPFPRLHFFMPGFAPLTSRGSQQYRALTVPELTQQMFDAKNMMAACDPRHGRYLTVAAVFRGRMSMKEVDEQMLSVQSKNSSYFVEWIPNNVKTAVCDIPPRGLKMAATFIGNSTAIQELFKRISEQFTAMFRRKAFLHWYTGEGMDEMEFTEAESNMNDLVSEYQQYQDATAEEGEFEEEAEEEVA

# Set the path to the pretrained VAE checkpoint
export vae_loadcheck_path=Path/to/pretrained_models/vae/opencell_finetuned.bin

# Specify the path to the pretrained model weights
export loadcheck_path=Path/to/pretrained_models/cell_diff/opencell_finetuned_all.bin

# Set the random seed
export seed=6

# Run the image generation script
bash scripts/cell_diff/evaluate_img_generation_opencell.sh