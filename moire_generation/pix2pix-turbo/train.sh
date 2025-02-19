# 1. accelerate config first!
export WORLD_SIZE=8
export MASTER_ADDR=hpe162
export MASTER_PORT=9887
export WANDB_API_KEY='1e349f15ef0ce18541da03a1ae74007c2add3e50'

# train_pix2pix_tb.py
accelerate launch train.py \
  --pretrained_model_name_or_path='stabilityai/sd-turbo' \
  --lora_rank_unet=8 \
  --lora_rank_vae=4 \
  --output_dir='/purestorage/project/tyk/3_CUProjects/FAS/pix2pix-turbo' \
  --seed=777 \
  --dataloader_num_workers=16 \
  --train_batch_size=4 \
  --num_training_epochs=1000 \
  --max_train_steps=10000 \
  --checkpointing_steps=500 \
  --allow_tf32 \
  --mixed_precision='no' \
  --deterministic \
  --resolution=1024 \
  --lambda_gan=2.0 \
  --lambda_l2=3.0 \