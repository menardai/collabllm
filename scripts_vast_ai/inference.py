#!/usr/bin/env python3
"""
Inference script for CollabLLM finetuned models.
Adapted from notebook_tutorials/inference_finetuned_model.ipynb

pip install torch transformers peft accelerate safetensors bitsandbytes

Example usage:
python inference.py --lora_adapter_path ../checkpoints/llama-3b-checkpoint --task_desc "Recommend a movie." --prompt "Find a film suitable for a date night."
"""

import argparse
import os
import sys
import logging
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig


def setup_logging():
    """Setup logging configuration."""
    logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
    
    # Check for CollabLLM logging environment variable
    enable_logging = os.environ.get("ENABLE_COLLABLLM_LOGGING", "0")
    if enable_logging == "0":
        logging.getLogger("collabllm").setLevel(logging.CRITICAL)


def load_model_and_tokenizer(lora_adapter_path):
    """Load the base model and LoRA adapter."""
    print(f"Loading LoRA adapter from: {lora_adapter_path}")
    
    # Load LoRA adapter configuration
    peft_config = PeftConfig.from_pretrained(lora_adapter_path)
    print(f"Base model for LoRA: {peft_config.base_model_name_or_path}")

    # Use the actual base model from the LoRA config
    actual_base_model = peft_config.base_model_name_or_path
    print(f"Using base model from adapter config: {actual_base_model}")

    # Load base model and tokenizer using the correct base model
    print(f"Loading base model: {actual_base_model}")
    tokenizer = AutoTokenizer.from_pretrained(actual_base_model, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        actual_base_model, 
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # Load LoRA adapter onto base model
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    print("LoRA adapter loaded successfully!")

    # Print model information
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded successfully!")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable percentage: {trainable_params/total_params:.2%}")

    # Set up tokenizer padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_response(model, tokenizer, messages, max_new_tokens=512, temperature=0.8):
    """Generate response from the model for given messages.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        messages: List of message dicts with 'role' and 'content' keys
                 OR a single string (will be treated as user message)
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature
    
    Returns:
        Generated response as string
    """
    # Handle backward compatibility with string input
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    
    # Apply chat template
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode response (only the generated part)
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return response.strip()


def run_inference_example(model, tokenizer, task_desc, user_prompt, max_new_tokens=512, temperature=0.8):
    """Run inference with system and user prompts."""
    print("=" * 60)
    print("GENERATING RESPONSE WITH SYSTEM + USER PROMPTS")
    print("=" * 60)
    print(f"System prompt: {task_desc}")
    print(f"User prompt: {user_prompt}")
    print("\n" + "-" * 60)

    # Create conversation with system and user messages
    messages = [
        {"role": "system", "content": task_desc},
        {"role": "user", "content": user_prompt}
    ]

    # Apply chat template
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    print("Formatted prompt:")
    print(formatted_prompt)
    print("\n" + "-" * 60)

    # Generate response
    print("Generating response...")
    response = generate_response(model, tokenizer, messages, max_new_tokens, temperature)

    print("\nModel Response:")
    print("=" * 60)
    print(response)
    print("=" * 60)

    return response


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CollabLLM finetuned model inference")
    
    parser.add_argument(
        "--lora_adapter_path",
        type=str,
        default="../checkpoints/llama-3b-checkpoint",
        help="Path to the LoRA adapter checkpoint"
    )
    parser.add_argument(
        "--task_desc",
        type=str,
        default="Recommend a movie.",
        help="Task description (system prompt)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Find a film that suitable for a date night. It should deliver an epic romantic drama, ideally in the 20th-century America, and carry the same decades-long, nostalgic storytelling spirit as Forrest Gump.",
        help="User prompt"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode for multiple queries"
    )
    
    return parser.parse_args()


def interactive_mode(model, tokenizer, task_desc):
    """Run the model in interactive mode."""
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("Enter your prompts below. Type 'quit' or 'exit' to stop.")
    print("System prompt:", task_desc)
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\nUser: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
                
            messages = [
                {"role": "system", "content": task_desc},
                {"role": "user", "content": user_input}
            ]
            
            print("Assistant: ", end="", flush=True)
            response = generate_response(model, tokenizer, messages)
            print(response)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main function."""
    args = parse_args()
    
    # Setup logging
    setup_logging()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.lora_adapter_path)
    print("Ready for inference!")
    
    if args.interactive:
        # Run in interactive mode
        interactive_mode(model, tokenizer, args.task_desc)
    else:
        # Run single inference
        response = run_inference_example(
            model, 
            tokenizer, 
            args.task_desc, 
            args.prompt,
            args.max_new_tokens,
            args.temperature
        )


if __name__ == "__main__":
    main()
