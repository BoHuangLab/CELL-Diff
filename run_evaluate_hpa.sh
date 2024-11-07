ulimit -c unlimited

[ -z "${save_dir}" ] && save_dir='./output/evaluate_hpa'

[ -z "${cell_image}" ] && cell_image='nucl,er,mt'
[ -z "${test_cell_image}" ] && test_cell_image='nucl,er,mt'
[ -z "${seed}" ] && seed=6

[ -z "${loadcheck_path}" ] && loadcheck_path=checkpoint.pt

# Dataset
[ -z "${data_path}" ] && data_path='dataset/HPA/test_lmdb_dataset'
[ -z "${img_crop_method}" ] && img_crop_method='center'
[ -z "${img_crop_size}" ] && img_crop_size=1024
[ -z "${img_resize}" ] && img_resize=256
[ -z "${split_key}" ] && split_key='test'

# DDPM
[ -z "${num_timesteps}" ] && num_timesteps=1000
[ -z "${ddpm_beta_start}" ] && ddpm_beta_start=0.0001
[ -z "${ddpm_beta_end}" ] && ddpm_beta_end=0.02
[ -z "${ddpm_schedule}" ] && ddpm_schedule='shifted_cos'
[ -z "${diffusion_pred_type}" ] && diffusion_pred_type='noise'

# Model
[ -z "${img_resize}" ] && img_resize=256
[ -z "${esm_embedding}" ] && esm_embedding='esm2'
[ -z "${embed_dim}" ] && embed_dim=640
[ -z "${max_protein_sequence_len}" ] && max_protein_sequence_len=2048
[ -z "${dit_patch_size}" ] && dit_patch_size=8
[ -z "${img_in_chans}" ] && img_in_chans=1
[ -z "${depth}" ] && depth=24
[ -z "${num_heads}" ] && num_heads=8
[ -z "${mlp_ratio}" ] && mlp_ratio=4
[ -z "${num_res_block}" ] && num_res_block='2,2,2'
[ -z "${dims}" ] && dims='64,128,256,512'
[ -z "${attn_drop}" ] && attn_drop=0.0

# Evaluation
[ -z "${timestep_respacing}" ] && timestep_respacing="ddim100"

python cell_diff/tasks/cell_diff/evaluate_hpa.py \
            --data_path $data_path \
            --img_crop_method $img_crop_method \
            --img_crop_size $img_crop_size \
            --img_resize $img_resize \
            --split_key $split_key \
            --cell_image $cell_image \
            --test_cell_image $test_cell_image \
            --img_resize $img_resize \
            --num_timesteps $num_timesteps \
            --ddpm_beta_start $ddpm_beta_start \
            --ddpm_beta_end $ddpm_beta_end \
            --ddpm_schedule $ddpm_schedule \
            --diffusion_pred_type $diffusion_pred_type \
            --esm_embedding $esm_embedding \
            --embed_dim $embed_dim \
            --max_protein_sequence_len $max_protein_sequence_len \
            --dit_patch_size $dit_patch_size \
            --img_in_chans $img_in_chans \
            --depth $depth \
            --num_heads $num_heads \
            --mlp_ratio $mlp_ratio \
            --num_res_block $num_res_block \
            --dims $dims \
            --attn_drop $attn_drop \
            --loadcheck_path $loadcheck_path \
            --seed 6 \
            --save_dir $save_dir \
            --timestep_respacing $timestep_respacing \
            --infer \

echo Compute FID-O
python -m pytorch_fid $save_dir/$split_key/compute_fid/generated_img/ $save_dir/$split_key/compute_fid/real_img/ --device cuda

echo Compute FID-T
python -m pytorch_fid $save_dir/$split_key/compute_fid/generated_threshold_img/ $save_dir/$split_key/compute_fid/real_threshold_img/ --device cuda
