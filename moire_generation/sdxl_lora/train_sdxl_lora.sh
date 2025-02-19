export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export DATASET_NAME="./uhdm2"
export OUTPUT_DIR="moire-SDXL-LoRA-exp5-sub1"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"
export WORLD_SIZE=1
export MASTER_ADDR=nv174
export MASTER_PORT=9887

accelerate launch train_dreambooth_lora_sdxl_advanced.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_PATH \
  --dataset_name=$DATASET_NAME \
  --instance_prompt="natural image in the style of TOK" \
  --validation_prompt="a TOK picture of an astronaut riding a horse with the TOK noise pattern" \
  --output_dir=$OUTPUT_DIR \
  --caption_column="prompt" \
  --mixed_precision="bf16" \
  --resolution=1024 \
  --train_batch_size=4 \
  --repeats=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1.0 \
  --text_encoder_lr=1.0 \
  --optimizer="prodigy" \
  --train_text_encoder_ti \
  --train_text_encoder_ti_frac=0.5 \
  --num_new_tokens_per_abstraction=2 \
  --snr_gamma=5.0 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --rank=8 \
  --max_train_steps=3000 \
  --checkpointing_steps=2000 \
  --seed="0" \
  --dataloader_num_workers=8 \
  --adam_weight_decay_text_encoder=0.0001 \
  --use_dora
  # --report_to="wandb"\
  # --image_column="file_name" \