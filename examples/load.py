"""
Utility: load a (possibly PEFT-adapted) causal-LM **and** its tokenizer
with sensible defaults for training or evaluation.

Highlights
----------
* 4-bit NF4 quantisation via `BitsAndBytesConfig` (flash-attention 2).
* Supports vanilla LM (`AutoModelForCausalLM`) **or**
  value-head LM (`AutoModelForCausalLMWithValueHead`).
* Transparently handles:
    • plain checkpoint  
    • PEFT directory (auto-wraps base model)  
    • optional PEFT config passed at runtime
* Prints trainable / total parameter counts.
"""

from __future__ import annotations

import os
import logging
from typing import Tuple, Type, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel, PeftConfig, get_peft_model
from trl import AutoModelForCausalLMWithValueHead

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# --------------------------------------------------------------------------- #
# Default 4-bit quantisation config                                           #
# --------------------------------------------------------------------------- #
DEFAULT_BNB_CFG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype=torch.bfloat16,
)


# --------------------------------------------------------------------------- #
# Main helper                                                                  #
# --------------------------------------------------------------------------- #
def load_model_and_tokenizer(
    model_name: str,
    *,
    max_new_tokens: int,
    is_eval: bool = True,
    model_class: Type[
        AutoModelForCausalLM | AutoModelForCausalLMWithValueHead
    ] = AutoModelForCausalLM,
    peft_config: Optional[PeftConfig] = None,
    bnb_config: BitsAndBytesConfig = DEFAULT_BNB_CFG,
    device: Optional[str] = None,
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    Parameters
    ----------
    model_name
        HF hub name or path to checkpoint / PEFT directory.
    max_new_tokens
        Unused inside but kept for API parity with caller.
    is_eval
        If `True`, model is returned in `.eval()` mode and tokenizer
        is set to `padding_side='left'`.
    model_class
        One of `AutoModelForCausalLM`, `AutoModelForCausalLMWithValueHead`.
    peft_config
        Optional PEFT config to wrap a base checkpoint.
    bnb_config
        4-bit quantisation settings (defaults to NF4).
    device
        Target device string, e.g. `"cuda:0"`.  If `None`, infer from
        `LOCAL_RANK` env-var.

    Returns
    -------
    model  : torch.nn.Module
    tokenizer : transformers.AutoTokenizer
    """
    # --------------------------------------------------------------------- #
    # Resolve device                                                        #
    # --------------------------------------------------------------------- #
    if device is None:
        local_rank = os.getenv("LOCAL_RANK", "0")
        device = f"cuda:{local_rank}"

    # --------------------------------------------------------------------- #
    # 1) Load checkpoint (PEFT dir or base model)                           #
    # --------------------------------------------------------------------- #
    if os.path.exists(model_name):
        logger.info("Loading PEFT model from local path %s", model_name)
        peft_cfg = PeftConfig.from_pretrained(model_name)

        if model_class is AutoModelForCausalLM:
            base = model_class.from_pretrained(
                peft_cfg.base_model_name_or_path,
                device_map={"": device},
                quantization_config=bnb_config,
            )
            model = PeftModel.from_pretrained(base, model_name, is_trainable=not is_eval)

        elif model_class is AutoModelForCausalLMWithValueHead:
            model = model_class.from_pretrained(
                model_name,
                device_map={"": device},
                quantization_config=bnb_config,
                is_trainable=not is_eval,
            )
        else:
            raise ValueError(f"Unsupported model_class: {model_class}")

        tokenizer = AutoTokenizer.from_pretrained(peft_cfg.base_model_name_or_path)

    else:
        logger.info("Loading %s from HF hub (%s)", model_class.__name__, model_name)

        if model_class is AutoModelForCausalLM:
            model = model_class.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_cache=not is_eval,
                device_map={"": device},
                quantization_config=bnb_config,
            )
            if peft_config is not None:
                model = get_peft_model(model, peft_config)

        elif model_class is AutoModelForCausalLMWithValueHead:
            if peft_config is None:
                raise ValueError("`peft_config` must be provided for value-head models.")
            model = model_class.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map={"": device},
                peft_config=peft_config,
                quantization_config=bnb_config,
                is_trainable=not is_eval,
            )
        else:
            raise ValueError(f"Unsupported model_class: {model_class}")

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # --------------------------------------------------------------------- #
    # 2) Tokenizer defaults                                                 #
    # --------------------------------------------------------------------- #
    tokenizer.padding_side = "left" if is_eval else "right"
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer padding side set to '%s'", tokenizer.padding_side)

    # --------------------------------------------------------------------- #
    # 3) Parameter statistics                                               #
    # --------------------------------------------------------------------- #
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * n_trainable / n_total
    logger.info("Trainable parameters: %d / %d (%.2f%%)", n_trainable, n_total, pct)

    return model.eval() if is_eval else model, tokenizer
