ulimit -c unlimited
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
[ -z "${output_dir}" ] && output_dir=pretrain_hpa/$WANDB_RUN_NAME

# Dataset
[ -z "${data_path}" ] && data_path=.
[ -z "${img_crop_method}" ] && img_crop_method=random
[ -z "${img_crop_size}" ] && img_crop_size=1024
[ -z "${img_resize}" ] && img_resize=256
[ -z "${seq_zero_mask_ratio}" ] && seq_zero_mask_ratio=0.5
[ -z "${cell_image}" ] && cell_image='nucl,er,mt'
[ -z "${split_key}" ] && split_key=train

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
[ -z "${cell_image_ratio}" ] && cell_image_ratio=0.5

# Loss
[ -z "${sequence_loss_coeff}" ] && sequence_loss_coeff=1.0
[ -z "${image_loss_coeff}" ] && image_loss_coeff=1.0

# Training
[ -z "${vae_loadcheck_path}" ] && vae_loadcheck_path=.
[ -z "${loadcheck_path}" ] && loadcheck_path=.
[ -z "${learning_rate}" ] && learning_rate=5e-5
[ -z "${weight_decay}" ] && weight_decay=0.0
[ -z "${gradient_accumulation_steps}" ] && gradient_accumulation_steps=4
[ -z "${per_device_train_batch_size}" ] && per_device_train_batch_size=16
[ -z "${per_device_eval_batch_size}" ] && per_device_eval_batch_size=128

[ -z "${num_train_epochs}" ] && num_train_epochs=5000
[ -z "${logging_dir}" ] && logging_dir=$output_dir
[ -z "${logging_steps}" ] && logging_steps=100
[ -z "${warmup_steps}" ] && warmup_steps=1000
[ -z "${max_steps}" ] && max_steps=100000
[ -z "${save_steps}" ] && save_steps=1000

[ -z "${MASTER_PORT}" ] && MASTER_PORT=29405
[ -z "${MASTER_ADDR}" ] && MASTER_ADDR=127.0.0.1

DISTRIBUTED_ARGS="--nproc_per_node $n_gpu \
                  --master_port $MASTER_PORT \
                  --master_addr $MASTER_ADDR"

python -m torch.distributed.run $DISTRIBUTED_ARGS cell_diff/tasks/cell_diff/pretrain_hpa.py \
            --output_dir $output_dir \
            --data_path $data_path \
            --img_crop_method $img_crop_method \
            --img_crop_size $img_crop_size \
            --img_resize $img_resize \
            --seq_zero_mask_ratio $seq_zero_mask_ratio \
            --cell_image $cell_image \
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
            --cell_image_ratio $cell_image_ratio \
            --sequence_loss_coeff $sequence_loss_coeff \
            --image_loss_coeff $image_loss_coeff \
            --vae_loadcheck_path $vae_loadcheck_path \
            --loadcheck_path $loadcheck_path \
            --learning_rate $learning_rate \
            --weight_decay $weight_decay \
            --gradient_accumulation_steps $gradient_accumulation_steps \
            --per_device_train_batch_size $per_device_train_batch_size \
            --per_device_eval_batch_size $per_device_eval_batch_size \
            --num_train_epochs $num_train_epochs \
            --logging_dir $logging_dir \
            --logging_steps $logging_steps \
            --warmup_steps $warmup_steps \
            --max_steps $max_steps \
            --save_steps $save_steps \
            --seed 666666 \

            # --ft \
            # --ifresume \
            # --bf16 \
            # --fp16 \