#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Llama 3.2 3B (Instruct) inference with Unsloth in 4-bit.

Usage (single turn):
  python infer_llama32_unsloth_4bit.py --prompt "Explain attention like I'm 5."

Interactive REPL:
  python infer_llama32_unsloth_4bit.py --repl

Common options:
  --model meta-llama/Llama-3.2-3B-Instruct
  --max-new-tokens 512
  --temperature 0.7
  --top-p 0.9
  --system "You are a helpful assistant."
  --flash-attn                # try flash-attention v2 if available
  --hf-token <YOUR_HF_TOKEN>  # or rely on environment/login

Tip: For CUDA memory fragmentation, you can export:
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"""

import os
import sys
import argparse
import torch
from unsloth import FastLanguageModel

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unsloth 4-bit inference for Llama 3.2 3B Instruct")
    p.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                   help="HF model ID")
    p.add_argument("--system", type=str, default="You are a helpful, concise assistant.",
                   help="System prompt")
    p.add_argument("--prompt", type=str, default=None, help="User prompt (omit to read from stdin once, or use --repl)")
    p.add_argument("--repl", action="store_true", help="Interactive chat loop (type Ctrl+D to exit)")
    p.add_argument("--max-seq-len", type=int, default=8192, help="Tokenizer/model max seq length hint")
    p.add_argument("--max-new-tokens", type=int, default=512, help="Max new tokens to generate")
    p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (0 = greedy)")
    p.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling")
    p.add_argument("--no-sampling", action="store_true", help="Disable sampling (greedy decode)")
    p.add_argument("--flash-attn", action="store_true", help="Use FlashAttention2 if available")
    p.add_argument("--hf-token", type=str, default=None, help="Hugging Face token (optional)")
    return p

def load_model(args):
    # Prefer bfloat16 on GPU; otherwise float32 on CPU.
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Choose attention backend
    attn_impl = "flash_attention_2" if args.flash_attn else "sdpa"

    print(f"[load] model={args.model} 4bit=True dtype={dtype} attn={attn_impl}", file=sys.stderr)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name       = args.model,
        max_seq_length   = args.max_seq_len,
        dtype            = dtype,            # Unsloth will set good defaults
        load_in_4bit     = True,            # 4-bit quantization via bitsandbytes (nf4 by default)
        device_map       = "auto",
        token            = args.hf_token,   # or None to use cached login/env
        attn_implementation = attn_impl,
        trust_remote_code   = False,
    )

    # Switch to inference mode (disables any training-specific hooks/adapters)
    FastLanguageModel.for_inference(model)

    # Ensure a valid pad token to silence warnings during generation
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer

def chat_once(model, tokenizer, system_prompt: str, user_prompt: str,
              max_new_tokens: int, temperature: float, top_p: float, no_sampling: bool) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    # Use the model's chat template when available
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if no_sampling or temperature <= 0.0:
        gen_kwargs.update(dict(do_sample=False))
    else:
        gen_kwargs.update(dict(
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
        ))

    with torch.no_grad():
        outputs = model.generate(input_ids=inputs, **gen_kwargs)

    # Slice off the prompt tokens to return only the new text
    gen_only = outputs[0, inputs.shape[-1]:]
    return tokenizer.decode(gen_only, skip_special_tokens=True)

def main():
    args = build_parser().parse_args()
    model, tokenizer = load_model(args)

    # Non-REPL single-run modes:
    if args.repl is False:
        # Priority: --prompt; else read one line from stdin if available; else error.
        user_prompt = args.prompt
        if user_prompt is None and not sys.stdin.isatty():
            user_prompt = sys.stdin.read().strip()
        if not user_prompt:
            print("No prompt provided. Use --prompt, pipe text on stdin, or run with --repl.", file=sys.stderr)
            sys.exit(2)

        out = chat_once(
            model, tokenizer,
            system_prompt=args.system,
            user_prompt=user_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            no_sampling=args.no_sampling,
        )
        print(out)
        return

    # Interactive REPL
    print(">> Llama 3.2 3B Instruct (Unsloth 4-bit) — type Ctrl+D to exit.")
    history = []
    while True:
        try:
            user_prompt = input("\nUser> ").strip()
        except EOFError:
            print("\nBye!")
            break
        if not user_prompt:
            continue

        # Keep a simple running context by appending last turns (optional).
        # For safety with small VRAM, we only keep the last system + 3 turns.
        history.append({"role": "user", "content": user_prompt})
        recent = history[-3:]  # last N user/assistant turns

        messages = [{"role": "system", "content": args.system}] + recent
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": not args.no_sampling and args.temperature > 0.0,
        }
        if gen_kwargs["do_sample"]:
            gen_kwargs.update(dict(temperature=float(args.temperature), top_p=float(args.top_p)))

        with torch.no_grad():
            outputs = model.generate(input_ids=inputs, **gen_kwargs)

        gen_only = outputs[0, inputs.shape[-1]:]
        assistant = tokenizer.decode(gen_only, skip_special_tokens=True)
        print(f"\nAssistant> {assistant}")
        history.append({"role": "assistant", "content": assistant})

if __name__ == "__main__":
    # Optional, helps with CUDA memory fragmentation on long runs.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
