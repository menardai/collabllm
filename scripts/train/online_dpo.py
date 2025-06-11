#!/usr/bin/env python3
"""
Preparation: To run the following, you need to generate multiturn data from `scripts/generate_reward_guided_multiturn_conv.py`

DPO train a causal-LM + LoRA adapter on a multi-turn dataset.
Example
-------
CUDA_VISIBLE_DEVICES=4 WANDB__SERVICE_WAIT=300 python -m scripts.train.online_dpo \

CUDA_VISIBLE_DEVICES=1,2 WANDB__SERVICE_WAIT=300 torchrun --master_port=56800 --nnodes=1 --nproc_per_node=2 -m scripts.train.online_dpo \
    --dataset_name math-hard \
    --metric_names "accuracy" "interactivity" "token_amount" \
    --metric_weights 1 1 -0.5 \
    --user_generation_kwargs '{"model": "gpt-4o-mini"}' \
    --assistant_generation_kwargs '{"model": "sft-Llama-3.1-8B-Instruct", "temperature": 0.6}' \
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
    --eval_steps 1 \
    --logging_steps 1 \
    --wandb_entity dsp-team \
    --wandb_project collabllm \
    --use_4bit
"""

from __future__ import annotations

import argparse, os, json
import wandb
import hashlib
from typing import Tuple, Optional
from dotenv import load_dotenv
import numpy as np

from trl import OnlineDPOConfig, OnlineDPOTrainer
from trl.trainer.judges import BasePairwiseJudge

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
from examples.metrics import *

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Parameter-free multiturn DPO trainer")

    # Data / paths
    p.add_argument("--dataset_repo", type=str, required=True)
    p.add_argument("--dataset_name", type=str, required=True)
    p.add_argument("--metric_names", nargs="+", required=True)
    p.add_argument("--user_generation_kwargs", type=json.loads, default="{}")
    p.add_argument("--assistant_generation_kwargs", type=json.loads, default="{}")
    p.add_argument("--reward_generation_kwargs", type=json.loads, default="{}")
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
    gpu_memory_utilization: float = 0.5,
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
        from typing import Type
        from vllm.config import VllmConfig
        from vllm.worker.worker import Worker
        from vllm.worker.model_runner import GPUModelRunnerBase

        ori_worker_init = Worker.__init__

        def patched_worker_init_for_custom_device(
                self,
                vllm_config: VllmConfig,
                local_rank: int,
                rank: int,
                distributed_init_method: str,
                is_driver_worker: bool = False,
                model_runner_cls: [Type[GPUModelRunnerBase]] = None,
        ):
            new_local_rank = int(os.environ.get('LOCAL_RANK', 0))
            return ori_worker_init(self, vllm_config, new_local_rank, rank, distributed_init_method, is_driver_worker, model_runner_cls)

        Worker.__init__ = patched_worker_init_for_custom_device

        vllm_base_model = LLM(
            model=base_model_name,
            dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16",
            quantization="bitsandbytes" if bnb_cfg else None,
            load_format="bitsandbytes" if bnb_cfg else None,
            enable_lora=True,
            max_lora_rank=lora_cfg.r,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=1 # torch.cuda.device_count()
        )
        os.environ["RANK"] = str(rank)
    except ImportError:
        vllm_base_model = None
    return model, tok, vllm_base_model

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

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
        is_eval=False,
    )

    # DeepSpeed zero
    ds_cfg = {
        "zero_optimization": {
            "stage": 2,
            "overlap_comm": False,
            "reduce_bucket_size": "auto",
            "contiguous_gradients": True,
            "offload_optimizer": {"device": "none"},
            "offload_param": {"device": "none"},
        },
        "gradient_clipping": "auto",
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "steps_per_print": 200,
    }

    # Trainer config
    train_args = OnlineDPOConfig(
        beta=0.1,
        loss_type="sigmoid",
        max_grad_norm=1.0,
        optim="adamw_torch",
        report_to="wandb",
        do_eval=False,
        eval_steps=args.eval_steps, 
        save_strategy='steps',
        save_steps=1,
        eval_strategy="no",
        gradient_checkpointing=True,  
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_train_epochs,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        max_new_tokens=args.max_new_tokens, 
        max_length=args.max_prompt_length + args.max_new_tokens, 
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        run_name=args.output_dir,
        output_dir=args.output_dir,
        deepspeed=ds_cfg, 
        use_vllm=False,
        fp16=not torch.cuda.is_bf16_supported(), 
        bf16=torch.cuda.is_bf16_supported(),
    )

    # W&B
    if args.wandb_project and os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.output_dir.replace("/", "_"),
            config=train_args.to_dict(),
            save_code=True,
            job_type="train",
        )

    ######################## JUDGE ########################
    def compute_hash(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()
    
    str_prompt_to_multiturn_data_map = {}
    def process(row):
        str_prompt = tok.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
        str_prompt_to_multiturn_data_map.setdefault(
            compute_hash(str_prompt), 
            {k: row[k] for k in ["single_turn_prompt", "single_turn_completion", "single_turn_metadata", "prompt"]}
        )
        row["prompt"] = str_prompt
        return row

    ds["train"] = ds["train"].map(process, load_from_cache_file=False)
    collabllm_model_kwargs = {
        "local_model": model,
        "local_tokenizer": tok,
        "vllm_base_model": vllm,
    }
    class MultiturnRewardJudge(BasePairwiseJudge):
        def judge(self, prompts, completion_pairs, shuffle_order=False):
            rank_of_the_first_completion = []
            for prompt, completion_pair in zip(prompts, completion_pairs):
                multiturn_data = str_prompt_to_multiturn_data_map.get(compute_hash(prompt))
                chat_histories = [
                    multiturn_data["prompt"] + 
                    [{"role": "assistant", "content": completion}] for completion in completion_pair
                ]
                pair_rewards = []
                for chat_history in chat_histories:
                    reward_info = multiturn_aware_reward(
                        chat_history=chat_history,
                        task_desc=datasets_info[args.dataset_name]["task_desc"],
                        single_turn_prompt=multiturn_data["single_turn_prompt"],
                        single_turn_completion=multiturn_data["single_turn_completion"],
                        metadata=multiturn_data["single_turn_metadata"],
                        metric_names=args.metric_names,
                        metric_weights=args.metric_weights,
                        user_generation_kwargs=args.user_generation_kwargs,
                        assistant_generation_kwargs=args.assistant_generation_kwargs,
                        reward_generation_kwargs=args.reward_generation_kwargs,
                        num_samples=args.num_samples,
                        max_new_turns=args.max_new_turns,
                        **collabllm_model_kwargs
                    )
                    pair_rewards.append(np.mean(reward_info["MR"]))
                rank_of_the_first_completion.append(np.argmax(pair_rewards).item())
                logger.info(
                    f"\n[Response 1] {completion_pair[0]}\n\n[Response 2] {completion_pair[1]}\nRewards: {pair_rewards}\n"
                )
            return torch.tensor(rank_of_the_first_completion)
            
                
    judge = MultiturnRewardJudge()  

    ######################## Trainer ########################
    trainer = OnlineDPOTrainer(
        model=model,
        judge=judge,
        train_dataset=ds["train"],
        processing_class=tok,
        args=train_args,
    )

    trainer.model.print_trainable_parameters()
    trainer.train(resume_from_checkpoint=args.resume_ckpt_dir)

    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

    if args.push_to_hub and args.hf_org:
        repo = f"offline_dpo-{args.dataset_repo.replace('/', '_')}"
        trainer.model.push_to_hub(f"{args.hf_org}/{repo}", private=True)
        tok.push_to_hub(f"{args.hf_org}/{repo}", private=True)

    if args.wandb_project:
        wandb.finish()

if __name__ == "__main__":
    load_dotenv(".env")

    main()