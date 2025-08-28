#!/usr/bin/env python3
"""
Preparation: To run the following, you need to generate multiturn data from `scripts.engine.build_dataset`

DPO train a causal-LM + LoRA adapter on a multi-turn dataset.
Example (on 4 NVIDIA A100-SXM4-80GB GPUs):
-------
CUDA_VISIBLE_DEVICES=0,1,2,3 WANDB__SERVICE_WAIT=300 torchrun --master_port=56500 --nnodes=1 --nproc_per_node=4 -m scripts.train.offline_dpo \
    --dataset_repo collabllm/collabllm-multiturn-medium \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --output_dir outputs/offline_dpo_from_base/collabllm-multiturn-medium \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --save_total_limit 10 \
    --num_train_epochs 8 \
    --learning_rate 5e-6 \
    --eval_steps 10 \
    --logging_steps 1 \
    --wandb_entity dsp-team \
    --wandb_project collabllm \
    --use_lora

CUDA_VISIBLE_DEVICES=0,1,2,3 WANDB__SERVICE_WAIT=300 torchrun --master_port=56500 --nnodes=1 --nproc_per_node=4 -m scripts.train.offline_dpo \
    --dataset_repo collabllm/collabllm-multiturn-math-hard \
    --model_name outputs/sft/collabllm-multiturn-math-hard \
    --output_dir outputs/offline_dpo/collabllm-multiturn-math-hard \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --save_total_limit 10 \
    --num_train_epochs 8 \
    --learning_rate 5e-6 \
    --eval_steps 10 \
    --logging_steps 1 \
    --wandb_entity dsp-team \
    --wandb_project collabllm \
    --use_lora
"""

from __future__ import annotations

import argparse, os, json
from typing import Tuple, Optional

os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Optional set GPU device ID

from unsloth import FastLanguageModel, PatchDPOTrainer
from unsloth import is_bfloat16_supported
PatchDPOTrainer()   # *** Unsloth ***

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftConfig, PeftModel, LoraConfig
from collabllm.datasets.multiturn import MultiturnDataset
from trl import DPOConfig, DPOTrainer
import wandb


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Parameter-free multiturn DPO trainer")

    # Data / paths
    p.add_argument("--dataset_repo", type=str, required=True)
    p.add_argument("--eval_ratio",   type=float, default=0.1)
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
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--warmup_ratio", type=float, default=0)        
    p.add_argument("--logging_steps", type=int, default=1)           
    p.add_argument("--max_prompt_length", type=int, default=4096) 
    p.add_argument("--max_new_tokens", type=int, default=2048) 
    p.add_argument("--minimum_gap", type=float, default=0.02) 

    # Precision / hardware
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use_lora", action="store_true", default=True)
    p.add_argument("--use_4bit", action="store_true", default=True)

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
    max_seq_length: int = 4096,
    is_eval: bool = False,
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    try:
        pc = PeftConfig.from_pretrained(model_name)
        base, tok = FastLanguageModel.from_pretrained(
            pc.base_model_name_or_path,
            max_seq_length = max_seq_length,
            dtype = None,
            load_in_4bit = True,
        )
        model = PeftModel.from_pretrained(base, model_name, is_trainable=not is_eval)
        # tok = AutoTokenizer.from_pretrained(pc.base_model_name_or_path, trust_remote_code=True)
    except Exception:
        model, tok = FastLanguageModel.from_pretrained(
            model_name,
            dtype = None,
            load_in_4bit = True,
            max_seq_length = max_seq_length,
        )
        # tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = FastLanguageModel.get_peft_model(
            model,
            r = 64,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 64,
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 3407,
            max_seq_length = max_seq_length,
        )

    tok.padding_side, tok.pad_token = ("left" if is_eval else "right"), tok.eos_token
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,}/{total:,} ({trainable/total:.2%})")
    # Required when using gradient checkpointing
    if hasattr(model, "config"):
        try:
            model.config.use_cache = False
        except Exception:
            pass
    return model, tok

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Dataset
    ds = MultiturnDataset(args.dataset_repo).to_dpo_dataset(eval_ratio=args.eval_ratio, minimum_gap=args.minimum_gap)

    # Bits-and-bytes
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=args.use_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    ) if args.use_4bit else None

    # LoRA
    lora_cfg = LoraConfig(
        r=args.peft_r,
        lora_alpha=args.peft_alpha,
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights="gaussian",
        target_modules=args.target_modules.split(","),
    ) if args.use_lora else None
    
    # Load model
    model, tok = load_model_and_tokenizer(
        args.model_name,
        bnb_cfg=bnb_cfg,
        lora_cfg=lora_cfg,
        device=args.device,
        max_seq_length=args.max_prompt_length,
        is_eval=False,
    )

    # Trainer config
    train_args = DPOConfig(
        beta=0.1,
        loss_type="sigmoid",
        max_grad_norm=1.0,
        report_to="wandb",
        do_eval=True,
        eval_steps=args.eval_steps, 
        save_strategy='epoch',
        eval_strategy="steps",
        lr_scheduler_type="cosine",
        metric_for_best_model="eval_loss",
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_train_epochs,
        save_total_limit=args.save_total_limit,
        max_length=args.max_new_tokens, 
        max_prompt_length=args.max_prompt_length, 
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        run_name=args.output_dir,
        output_dir=args.output_dir,
        
        gradient_checkpointing=True,  
        gradient_checkpointing_kwargs={'use_reentrant': False},
        precompute_ref_log_probs=True,
        # Disable length-based grouping since our dataset items are not tokenized
        # and thus do not contain 'input_ids' for automatic length inference.
        group_by_length=False,

        optim="adamw_8bit",     # adamw_torch
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
    )

    # W&B
    if args.wandb_project and os.environ.get("RANK", "0") == "0":
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.output_dir.replace("/", "_"),
            config=train_args.to_dict(),
            save_code=True,
            job_type="train",
        )
    
    def process(row):
        reference = tok.apply_chat_template(row["prompt"] + [{'role': 'assistant', 'content': row["chosen"]}], tokenize=False)
        row["prompt"] = tok.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
        row["chosen"] = row["chosen"].strip() + tok.eos_token
        row["rejected"] = row["rejected"].strip() +  tok.eos_token
        assert row["prompt"] + row["chosen"] == reference
        return row

    # ATTENTION: TODO --> tok.eos_token is also used in load_model_and_tokenizer()
    def process_non_llama(row):
        messages = row["prompt"]
        prompt_str = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

        chosen_body = row["chosen"].strip()
        rejected_body = row["rejected"].strip()

        ref_chosen = tok.apply_chat_template(messages + [{'role': 'assistant', 'content': chosen_body}], tokenize=False, enable_thinking=False)
        ref_rejected = tok.apply_chat_template(messages + [{'role': 'assistant', 'content': rejected_body}], tokenize=False, enable_thinking=False)

        row["prompt"] = prompt_str
        row["chosen"] = ref_chosen[len(prompt_str):]
        row["rejected"] = ref_rejected[len(prompt_str):]

        assert row["prompt"] + row["chosen"] == ref_chosen
        return row

    if args.model_name.startswith("meta-llama/"):
        ds["train"] = ds["train"].map(process, load_from_cache_file=False)
        ds["eval"] = ds["eval"].map(process, load_from_cache_file=False)
    else:
        print("NON-Llama model detected, using custom processing")
        ds["train"] = ds["train"].map(process_non_llama, load_from_cache_file=False)
        ds["eval"] = ds["eval"].map(process_non_llama, load_from_cache_file=False)
        print("Custom processing complete")

    trainer = DPOTrainer(
        model=model,
        ref_model = None,   # unsloth
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        processing_class=tok,
        peft_config=None,  # Disable peft_config since we already applied LoRA via FastLanguageModel
        args=train_args,
    )
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
    main()