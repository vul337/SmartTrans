#!/bin/bash

accelerate launch --config_file default_config.yaml pretrain.py \
    --train_file "./pretrain_dataset/train_datas_512.txt" \
    --validation_file "./pretrain_dataset/eval_datas_512.txt" \
    --tokenizer_name "./tokenizer" \
    --learning_rate 1e-4 \
    --weight_decay 1e-2 \
    --num_train_epochs 4 \
    --gradient_accumulation_steps 16 \
    --model_type "roformer" \
    --pad_to_max_length True \
    --max_seq_length 512 \
    --line_by_line True \
    --mlm_probability 0.15 \
    --output_dir "./model/" \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 256 \
    --preprocessing_num_workers 4 \
    --report_to "wandb" \
    --with_tracking \
    --checkpointing_steps "epoch" \
    --num_warmup_steps 1250 \
    --lr_scheduler_type "linear" \
    --config_name "./config.json"

