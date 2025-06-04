"""
Multi-turn reward computation (one call to ChatSessionSimulator).

Assumes:
• ChatSessionSimulator.run_chat_simulation now accepts `num_samples`
  and returns a list of conversations in one shot (internally parallel/batched).
• SingleTurnOrChatMetric unchanged.
"""

from __future__ import annotations

import logging
import statistics as stats
from typing import Any, Dict, List, Sequence, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoModelForCausalLM, AutoTokenizer

from collabllm.metric import SingleTurnOrChatMetric
from collabllm.simulation import ChatSessionSimulator
from collabllm.utils.template import strip_system_prompt


logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Metric helper                                                               #
# --------------------------------------------------------------------------- #
def _score_one_metric(
    metric_name: str,
    messages: List[Dict[str, str]],
    metric_kwargs: Dict[str, Any],
    prompt: str,
    completion: str,
    metadata: Dict[str, Any] | None,
) -> float:
    metric = SingleTurnOrChatMetric(signature=metric_name, **metric_kwargs)
    return metric(
        messages=messages,
        single_turn_prompt=prompt,
        single_turn_completion=completion,
        metadata=metadata,
    )

# --------------------------------------------------------------------------- #
# Helper: pretty summary table                                                #
# --------------------------------------------------------------------------- #
def _log_reward_summary(reward_dict: Dict[str, List[float]]) -> None:
    """Compute mean / std for each metric list in `reward_dict` and log."""
    rows = []
    for metric, vals in reward_dict.items():
        # vals is always a list after evaluation, including "MR"
        mu = stats.mean(vals)
        sd = stats.stdev(vals) if len(vals) > 1 else 0.0
        rows.append((metric, f"{mu:.3f}", f"{sd:.3f}"))

    header = ("Metric", "Mean", "Std")

    try:
        from tabulate import tabulate
        table = "\n" + tabulate(rows, headers=header, tablefmt="github")
    except ImportError:
        colw = [max(len(x) for x in col) for col in zip(*([header] + rows))]
        fmt = "  ".join(f"{{:<{w}}}" for w in colw)
        table = "\n" + fmt.format(*header) + "\n" + "\n".join(fmt.format(*r) for r in rows)

    logger.info("Reward statistics:%s", table)

# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
def multiturn_aware_reward(
    *,
    task_description: str,  
    single_turn_prompt: str,
    single_turn_completion: str,
    metric_names: Sequence[str],
    reward_generation_kwargs: Dict[str, Any] | None = None,
    metadata: Dict[str, Any] | None = None,
    metric_weights: Sequence[float] | None = None,
    max_metric_workers: int = 16,
    return_details: bool = False,
    **chat_simulation_kwargs
) -> Dict[str, Any]:
    """
    Compute rewards for `num_samples` conversations returned in one batch.
    """
    reward_generation_kwargs = reward_generation_kwargs or {}
    metric_weights = metric_weights or [1.0] * len(metric_names)
    if len(metric_weights) != len(metric_names):
        raise ValueError("`metric_weights` length must equal `metric_names` length")

    # ------------------------------------------------------------------ #
    # 1 · Generate all conversations in one call                         #
    # ------------------------------------------------------------------ #
    sessions = ChatSessionSimulator().run_chat_simulation(
        task_description=task_description,
        single_turn_prompt=single_turn_prompt,
        **chat_simulation_kwargs

    )  # → List[List[dict]]
    # strip system message, if any
    sessions = [strip_system_prompt(session) for session in sessions]

    # ------------------------------------------------------------------ #
    # 2 · Prepare result containers                                      #
    # ------------------------------------------------------------------ #
    reward_dict: Dict[str, List[float]] = {m: [] for m in metric_names}
    reward_dict["MR"] = []

    # ------------------------------------------------------------------ #
    # 3 · Metric evaluation (fully parallel over conv × metric)          #
    # ------------------------------------------------------------------ #
    n_conv = len(sessions)
    # initialise storage
    for m in metric_names:
        reward_dict[m] = [0.0] * n_conv
    reward_dict["MR"] = [0.0] * n_conv
    
    with ThreadPoolExecutor(max_workers=max_metric_workers) as pool:
        fut_to_ctx = {}
        for conv_idx, messages in enumerate(sessions):
            for i, metric_name in enumerate(metric_names):
                fut = pool.submit(
                    _score_one_metric,
                    metric_name,
                    messages,
                    reward_generation_kwargs,
                    single_turn_prompt,
                    single_turn_completion,
                    metadata,
                )
                # keep context: which conversation / which metric / weight index
                fut_to_ctx[fut] = (conv_idx, i, metric_name)

        for fut in as_completed(fut_to_ctx):
            conv_idx, i, metric_name = fut_to_ctx[fut]
            score = fut.result()
            reward_dict[metric_name][conv_idx] = score

    # ------------------------------------------------------------------ #
    # 4 · Aggregate  →  Multiturn-aware Reward (MR)                       #
    # ------------------------------------------------------------------ #
    for conv_idx in range(n_conv):
        reward_dict["MR"][conv_idx] = sum(
            reward_dict[m][conv_idx] * metric_weights[i]
            for i, m in enumerate(metric_names)
        )
    _log_reward_summary(reward_dict)
    if return_details:
        return reward_dict, sessions
    return reward_dict


def parallel_multiturn_aware_reward(
    chat_histories: List[List[Dict[str, str]]],
    *,
    base_sim_args: Dict[str, Any],
    single_turn_completion: str,
    metric_names: List[str],
    reward_generation_kwargs: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    metric_weights: Optional[List[float]] = None,
    max_new_turns: int,
    num_samples: int,
    max_workers: int,
    max_metric_workers: int,
    return_details: bool = True,
    verbose: bool = False,
) -> List[Tuple[Any, Any]]:
    """
    Run multiturn_aware_reward in parallel over multiple chat histories.
    Asserts that 'local_model' and 'vllm_base_model' in base_sim_args are None.
    """
    # Ensure no local_model or vllm_base_model are used
    assert base_sim_args.get("local_model") is None, "local_model must be None"
    assert base_sim_args.get("vllm_base_model") is None, "vllm_base_model must be None"

    reward_generation_kwargs = reward_generation_kwargs or {}
    metric_weights = metric_weights or [1.0] * len(metric_names)

    def _evaluate_single(idx: int, history: List[Dict[str, str]]):
        res = multiturn_aware_reward(
            **base_sim_args,
            single_turn_completion=single_turn_completion,
            metric_names=metric_names,
            reward_generation_kwargs=reward_generation_kwargs,
            metadata=metadata,
            metric_weights=metric_weights,
            chat_history=history,
            max_new_turns=max_new_turns,
            num_samples=num_samples,
            max_workers=max_workers,
            max_metric_workers=max_metric_workers,
            return_details=return_details,
            verbose=verbose,
        )
        return idx, res

    results: List[Optional[Tuple[Any, Any]]] = [None] * len(chat_histories)
    worker_count = min(len(chat_histories), max_workers)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_evaluate_single, idx, hist): idx
            for idx, hist in enumerate(chat_histories)
        }
        for fut in as_completed(futures):
            idx, res = fut.result()
            results[idx] = res  # type: ignore

    if not return_details:
        return [(r, None) for r in results]  # type: ignore

    return results  # type: ignore
