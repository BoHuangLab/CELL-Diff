ulimit -c unlimited
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
[ -z "${output_dir}" ] && output_dir=.

# Dataset
[ -z "${data_path}" ] && data_path=.
[ -z "${img_crop_method}" ] && img_crop_method=center
[ -z "${img_crop_size}" ] && img_crop_size=1024
[ -z "${img_resize}" ] && img_resize=256
[ -z "${cell_image}" ] && cell_image='nucl,er,mt'
[ -z "${test_cell_image}" ] && test_cell_image='nucl,er,mt'
[ -z "${split_key}" ] && split_key=test

# DDPM
[ -z "${num_timesteps}" ] && num_timesteps=200
[ -z "${ddpm_beta_start}" ] && ddpm_beta_start=0.0001
[ -z "${ddpm_beta_end}" ] && ddpm_beta_end=0.02
[ -z "${ddpm_schedule}" ] && ddpm_schedule=squaredcos_cap_v2
[ -z "${diffusion_pred_type}" ] && diffusion_pred_type=noise

# Model
## VAE
[ -z "${num_down_blocks}" ] && num_down_blocks=3
[ -z "${latent_channels}" ] && latent_channels=4
[ -z "${vae_block_out_channels}" ] && vae_block_out_channels='128,256,512'

## CELL-Diff
[ -z "${block_out_channels}" ] && block_out_channels='320,640,1280,1280'
[ -z "${layers_per_block}" ] && layers_per_block=2
[ -z "${mid_num_attention_heads}" ] && mid_num_attention_heads=8
[ -z "${sample_size}" ] && sample_size=64
[ -z "${esm_embedding}" ] && esm_embedding=esm2
[ -z "${hidden_size}" ] && hidden_size=1280
[ -z "${max_protein_sequence_len}" ] && max_protein_sequence_len=2048
[ -z "${num_hidden_layers}" ] && num_hidden_layers=8
[ -z "${num_attention_heads}" ] && num_attention_heads=8
[ -z "${mlp_ratio}" ] && mlp_ratio=4
[ -z "${attn_drop}" ] && attn_drop=0.0
[ -z "${dit_patch_size}" ] && dit_patch_size=1

# Training
[ -z "${vae_loadcheck_path}" ] && vae_loadcheck_path=.
[ -z "${loadcheck_path}" ] && loadcheck_path=.

# Evaluation
[ -z "${timestep_respacing}" ] && timestep_respacing=ddim100


python cell_diff/tasks/cell_diff/eval_hpa.py \
            --output_dir $output_dir \
            --data_path $data_path \
            --img_crop_method $img_crop_method \
            --img_crop_size $img_crop_size \
            --img_resize $img_resize \
            --cell_image $cell_image \
            --test_cell_image $test_cell_image \
            --split_key $split_key \
            --num_timesteps $num_timesteps \
            --ddpm_beta_start $ddpm_beta_start \
            --ddpm_beta_end $ddpm_beta_end \
            --ddpm_schedule $ddpm_schedule \
            --diffusion_pred_type $diffusion_pred_type \
            --num_down_blocks $num_down_blocks \
            --latent_channels $latent_channels \
            --vae_block_out_channels $vae_block_out_channels \
            --block_out_channels $block_out_channels \
            --layers_per_block $layers_per_block \
            --mid_num_attention_heads $mid_num_attention_heads \
            --sample_size $sample_size \
            --esm_embedding $esm_embedding \
            --hidden_size $hidden_size \
            --max_protein_sequence_len $max_protein_sequence_len \
            --num_hidden_layers $num_hidden_layers \
            --num_attention_heads $num_attention_heads \
            --mlp_ratio $mlp_ratio \
            --attn_drop $attn_drop \
            --dit_patch_size $dit_patch_size \
            --vae_loadcheck_path $vae_loadcheck_path \
            --loadcheck_path $loadcheck_path \
            --timestep_respacing $timestep_respacing \
            --seed 6 \
            --infer \


echo Compute FID-O
python -m pytorch_fid $output_dir/$split_key/compute_fid/generated_img/ $output_dir/$split_key/compute_fid/real_img/ --device cuda

echo Compute FID-T
python -m pytorch_fid $output_dir/$split_key/compute_fid/generated_threshold_img/ $output_dir/$split_key/compute_fid/real_threshold_img/ --device cuda