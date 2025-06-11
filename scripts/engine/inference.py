#!/usr/bin/env python3
"""
Preparation: To run the following, you need to generate multiturn data from `scripts/generate_reward_guided_multiturn_conv.py`

DPO train a causal-LM + LoRA adapter on a multi-turn dataset.
Example (on 4 NVIDIA A100-SXM4-80GB GPUs):
-------
ENABLE_COLLABLLM_LOGGING=0 LLM_USE_V1=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=1,2,3,4,6,7 \
    torchrun --master_port=56500 --nnodes=1 --nproc_per_node=6 -m scripts.train.online_dpo \
    --dataset_name math-hard \
    --metric_names "accuracy" "interactivity" "token_amount" \
    --metric_weights 1 1 -0.5 \
    --user_generation_kwargs '{"model": "gpt-4o-mini"}' \
    --assistant_generation_kwargs '{"model": "sft-math-hard-Llama-3.1-8B-Instruct", "temperature": 0.6}' \
    --reward_generation_kwargs '{"model": "claude-3-5-sonnet-latest"}' \
    --dataset_repo collabllm/collabllm-multiturn-math-hard \
    --model_name outputs/sft/collabllm-multiturn-math-hard \
    --output_dir outputs/online_dpo/collabllm-multiturn-math-hard \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --save_total_limit 10 \
    --num_train_epochs 8 \
    --learning_rate 5e-6 \
    --gpu_memory_utilization 0.8 \
    --eval_steps 1 \
    --logging_steps 1 \
    --wandb_entity dsp-team \
    --wandb_project collabllm \
    --use_4bit
"""
from __future__ import annotations

import argparse, os, json
import torch.distributed as dist
import wandb
import hashlib
from typing import Tuple, Optional
from dotenv import load_dotenv
from datetime import timedelta
import numpy as np
import copy

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftConfig, PeftModel, LoraConfig, get_peft_model
from collabllm.datasets.multiturn import MultiturnDataset
from collabllm.reward import multiturn_aware_reward
from examples.single_turn_ds import datasets_info
from collabllm.simulation import ChatSessionSimulator
from examples.metrics import *

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Parameter-free multiturn DPO trainer")

    # Data / paths
    p.add_argument("--dataset_repo", type=str, required=True)
    p.add_argument("--dataset_name", type=str, required=True)
    p.add_argument("--eval_metric_names", nargs="+", required=True)
    p.add_argument("--user_generation_kwargs", type=json.loads, default="{}")
    p.add_argument("--assistant_generation_kwargs", type=json.loads, default="{}")
    p.add_argument("--eval_generation_kwargs", type=json.loads, default="{}")
    p.add_argument("--metric_weights", type=float, nargs="+", default=None)
    p.add_argument("--max_new_turns", type=int, default=4)
    p.add_argument("--num_samples", type=int, default=3)

    p.add_argument("--eval_ratio",   type=float, default=0)
    p.add_argument("--output_dir",   type=str, required=True)
    p.add_argument("--resume_ckpt_dir", type=str, default=None)

    # Base / adapter models
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--peft_r",     type=int,   default=32)
    p.add_argument("--peft_alpha", type=int,   default=16)
    p.add_argument("--peft_dropout", type=float, default=0.1)
    p.add_argument("--target_modules",
                   type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # Optim & schedule
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--warmup_ratio", type=float, default=0)        
    p.add_argument("--logging_steps", type=int, default=1)           
    p.add_argument("--max_prompt_length", type=int, default=2048) 
    p.add_argument("--max_new_tokens", type=int, default=2048) 
    p.add_argument("--minimum_gap", type=float, default=0.1) 
    
    # Precision / hardware
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use_4bit", action="store_true")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.6)

    # Tracking
    p.add_argument("--wandb_project", type=str)
    p.add_argument("--wandb_entity",  type=str)
    p.add_argument("--push_to_hub",   action="store_true")
    p.add_argument("--hf_org",        type=str)

    # Optional JSON/YAML override
    p.add_argument("--config_file", type=str)

    args = p.parse_args()
    if args.config_file:
        with open(args.config_file) as f:
            override = json.load(f) if args.config_file.endswith(".json") else \
                       __import__("yaml").safe_load(f)
        for k, v in override.items():
            setattr(args, k, v)
    return args

# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def load_model_and_tokenizer(
    model_name: str,
    bnb_cfg: Optional[BitsAndBytesConfig],
    lora_cfg: Optional[LoraConfig],
    device: str = "cuda",
    is_eval: bool = False,
    gpu_memory_utilization: float = 0.6,
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    try:
        pc = PeftConfig.from_pretrained(model_name)
        base = AutoModelForCausalLM.from_pretrained(
            pc.base_model_name_or_path,
            device_map={"": device},
            quantization_config=bnb_cfg,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_name, is_trainable=not is_eval)
        tok = AutoTokenizer.from_pretrained(pc.base_model_name_or_path, trust_remote_code=True)
        base_model_name = pc.base_model_name_or_path
    except Exception:
        logger.error(f"Failed to load PeftConfig for {model_name}, loading as base model.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": device},
            quantization_config=bnb_cfg,
            trust_remote_code=True,
        )
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if lora_cfg:
            model = get_peft_model(model, lora_cfg)
        base_model_name = model_name

    tok.padding_side, tok.pad_token = ("left" if is_eval else "right"), tok.eos_token
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,}/{total:,} ({trainable/total:.2%})")

    try:
        from vllm import LLM

        vllm_base_model = LLM(
            model=base_model_name,
            dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16",
            quantization="bitsandbytes" if bnb_cfg else None,
            enable_lora=True if lora_cfg else False,
            max_lora_rank=lora_cfg.r if lora_cfg else None,
            # Use `distributed_executor_backend="external_launcher"` so that
            # this llm engine/instance only creates one worker.
            distributed_executor_backend="external_launcher",
            gpu_memory_utilization=gpu_memory_utilization
        )
    except ImportError:
        vllm_base_model = None
    return model, tok, vllm_base_model

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    dataset_cls = datasets_info[args.dataset_name]["class"]
    task_desc = datasets_info[args.dataset_name]["task_desc"]
    dataset = dataset_cls().to_hf_dataset()

    testset = (
            dataset["test"].select(range(args.test_size))
            if args.test_size > 0
            else dataset["test"]
    )

    # Imporant for initializing vllm base model per GPU
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(backend='nccl', init_method=None)
    torch.cuda.set_device(local_rank)
    dist.barrier()

    # Dataset
    ds = MultiturnDataset(args.dataset_repo).to_inputs_dataset(eval_ratio=args.eval_ratio)

    # Bits-and-bytes
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=args.use_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=torch.bfloat16,
    ) if args.use_4bit else None

    # LoRA
    lora_cfg = LoraConfig(
        r=args.peft_r,
        lora_alpha=args.peft_alpha,
        lora_dropout=args.peft_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights="gaussian",
        target_modules=args.target_modules.split(","),
    )

    # Load model
    model, tok, vllm = load_model_and_tokenizer(
        args.model_name,
        bnb_cfg=bnb_cfg,
        lora_cfg=lora_cfg,
        device=args.device,
        is_eval=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    reward_dict = {m: [] for m in args.eval_metric_names}
    for example in testset:
        single_turn_prompt = example["single_turn_prompt"]
        single_turn_completion = example["single_turn_completion"]
        single_turn_metadata = example["single_turn_metadata"]

        base_sim_args = {
                "task_desc": task_desc,
                "single_turn_prompt": single_turn_prompt,
                "local_model": model,
                "local_tokenizer": tok,
                "vllm_base_model": vllm,
                "assistant_generation_kwargs": assistant_generation_kwargs,
                "user_generation_kwargs": user_generation_kwargs,
            }
        candidate_hist = ChatSessionSimulator().run_chat_simulation(
                **base_sim_args,
                num_samples=1,
                chat_history=chat_history,
                add_system_prompt_ratio=1.0,
                max_workers=args.max_workers,
                max_new_turns=args.max_new_turns,
                verbose=False,
            )

        rewards = multiturn_aware_reward(
                **base_sim_args,
                single_turn_completion=single_turn_completion,
                metadata=single_turn_metadata,
                metric_names=args.eval_metric_names,
                reward_generation_kwargs=args.eval_generation_kwargs,
                chat_history=candidate_hist,
                metric_weights=[0,0,0], # disregard metric weights
                max_new_turns=0, # no new turns, just evaluate the candidate
                num_samples=1, 
                max_workers=args.max_workers,
                max_metric_workers=args.max_metric_workers
            )
        for metric_name in args.eval_metric_names:
            reward_dict[metric_name].append(np.mean(rewards[metric_name]))
        
    # Save results
    for metric_name in args.eval_metric_names:
        reward_dict[metric_name] = np.mean(reward_dict[metric_name])
    
    output_path = os.path.join(args.output_dir, "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(reward_dict, f, indent=2)


if __name__ == "__main__":
    load_dotenv(".env")
    main()
    