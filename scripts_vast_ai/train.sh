#! /bin/bash
cd /data/collabllm
export OPENAI_API_KEY=fake_key
export WANDB_MODE=disabled
CUDA_VISIBLE_DEVICES=0 WANDB__SERVICE_WAIT=300 torchrun --master_port=56500 --nnodes=1 --nproc_per_node=1 -m scripts.train.offline_dpo \
    --dataset_repo collabllm/collabllm-multiturn-medium \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --output_dir outputs/offline_dpo_from_base/collabllm-multiturn-medium \ 
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --save_total_limit 10 \
    --num_train_epochs 8 \
    --learning_rate 5e-6 \
    --eval_steps 10 \
    --logging_steps 1 \
    --wandb_entity stephanemenard211 \
    --wandb_project collabllm \
    --use_lora