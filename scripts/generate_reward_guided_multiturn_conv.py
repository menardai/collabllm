import argparse
import json
import os.path as osp
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from dotenv import load_dotenv

from examples.single_turn_ds import datasets_info
from collabllm.datasets.multiturn import MultiturnDataset
from collabllm.synthetic import generate_metric_based_synthetic_conversation
from examples.metrics import *


def main(args):
    # Load single-turn dataset
    dataset_class = datasets_info[args.dataset_name]['class']
    task_description = datasets_info[args.dataset_name]['task_description']
    dataset = dataset_class().to_hf_dataset()

    multiturn_data_lst = []
    trainset = dataset['train'].select(range(args.conv_size)) if args.conv_size > 0 else dataset['train']

    for example in tqdm(trainset, desc="Generating multi-turn conversations"):
        multiturn_data = generate_metric_based_synthetic_conversation(
            task_description=task_description,
            single_turn_prompt=example['single_turn_prompt'],
            single_turn_completion=example['single_turn_completion'],
            single_turn_metadata=example['single_turn_metadata'],
            metric_names=args.metric_names,
            user_generation_kwargs=args.user_generation_kwargs,
            assistant_generation_kwargs=args.assistant_generation_kwargs,
            reward_generation_kwargs=args.reward_generation_kwargs,
            metric_weights=args.metric_weights,
            proact_prompt_ratio=args.proact_prompt_ratio,
            num_candidate_responses=args.num_candidate_responses,
            max_total_turns=args.max_total_turns,
            max_new_turns=args.max_new_turns,
            num_samples=args.num_samples,
            max_workers=args.max_workers,
            max_metric_workers=args.max_metric_workers,
            add_system_prompt_ratio=args.add_system_prompt_ratio
        )
        multiturn_data_lst.append(multiturn_data)

        # Save results to JSON
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = osp.join(args.output_dir, f"{args.dataset_name}_multiturn.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(multiturn_data_lst, f, indent=2)

        # Push to Hugging Face Hub
        MultiturnDataset(multiturn_data_lst).push_to_hub(
            repo_id=f"{args.hf_entity}/collabllm-multiturn-{args.dataset_name}"
        )

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multi-turn synthetic conversations with metrics.")

    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the single-turn dataset.")
    parser.add_argument("--metric_names", nargs="+", required=True, help="List of evaluation metric names.")
    parser.add_argument("--user_generation_kwargs", type=json.loads, default="{}", help="JSON dict of generation kwargs for user.")
    parser.add_argument("--assistant_generation_kwargs", type=json.loads, default="{}", help="JSON dict of generation kwargs for assistant.")
    parser.add_argument("--reward_generation_kwargs", type=json.loads, default="{}", help="Optional JSON dict for reward generation.")
    parser.add_argument("--metric_weights", type=float, nargs="+", default=None, help="Optional weights for each metric.")
    parser.add_argument("--proact_prompt_ratio", type=float, default=0.5, help="0 for none, 1 for proact, 0~1 for mixed.")
    parser.add_argument("--add_system_prompt_ratio", type=float, default=0, help="0 for none, 1 for proact, 0~1 for mixed.")
    parser.add_argument("--num_candidate_responses", type=int, default=2, help="Number of assistant candidates per turn.")
    parser.add_argument("--max_total_turns", type=int, default=14, help="Maximum number of conversation turns.")
    parser.add_argument("--max_new_turns", type=int, default=4, help="Window size for context in multi-turn generation.")
    parser.add_argument("--num_samples", type=int, default=3, help="Sample size for generating multiple conversations in one batch.")
    parser.add_argument("--conv_size", type=int, default=500, help="Number of conversations to generate.")
    parser.add_argument("--max_workers", type=int, default=16, help="Maximum number of parallel workers for sampling conversations.")
    parser.add_argument("--max_metric_workers", type=int, default=16, help="Maximum number of parallel workers for metrics.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated output.")
    parser.add_argument("--hf_entity", type=str, required=True, help="Hugging Face user or organization for dataset upload.")
    parser.add_argument("--save_steps", type=int, default=10, help="Save intermediate results every N steps.")

    load_dotenv(".env")

    args = parser.parse_args()

    print(args)
    main(args)

    # python -m scripts.generate_reward_guided_multiturn_conv   --dataset_name math-hard   --metric_names "accuracy" "interactivity" "token_amount"   --metric_weights 1 1 -0.5   --user_generation_kwargs '{"model": "gpt-4o-mini"}'   --assistant_generation_kwargs '{"model": "gpt-4o", "temperature": 0.6}'   --output_dir outputs/multiturn_conv   --hf_entity collabllm --reward_generation_kwargs '{"model": "claude-3-5-sonnet-latest"}'