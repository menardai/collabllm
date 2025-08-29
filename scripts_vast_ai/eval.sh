#! /bin/bash
cd /data/collabllm
export OPENAI_API_KEY=fake_key
export WANDB_MODE=disabled
export HF_HOME="/data/.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=1
CUDA_VISIBLE_DEVICES=0 WANDB__SERVICE_WAIT=300 torchrun --master_port=56500 --nnodes=1 --nproc_per_node=1 -m scripts.engine.inference_unsloth \
    --dataset_name medium \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --output_dir outputs/base/collabllm-multiturn-medium/inference-user-5mini \
    --eval_metric_names "document->bleu" "interactivity" "token_amount" \
    --user_generation_kwargs '{"model": "gpt-5-mini"}' \
    --assistant_generation_kwargs '{"model": "meta-llama/Llama-3.2-3B-Instruct", "temperature": 0.8}' \
    --eval_generation_kwargs '{"model": "gpt-5-mini"}' \
    --eval_size 20 \
    --use_lora \
    --use_4bit \
    --gpu_memory_utilization 0.5 \
    --max_model_len 4096 \
    --max_new_tokens 1024