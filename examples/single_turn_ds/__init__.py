import copy
import random
import re
import numpy as np
from typing import List

from datasets import load_dataset, Dataset, DatasetDict, interleave_datasets
from collabllm.prompts import SYSTEM_PROMPT

from .abg_coqa import AbgCoQA
from .math_hard import MATH
from .medium import Medium
from .bigcodebench import BigCodeBench


# ADD NEW DATASET BELOW
datasets_info = {
    'math-hard': {
        'task_description': 'question answering',
        'class': MATH
    },
    'abg-coqa': {
        'task_description': 'question answering',
        'class': AbgCoQA
    },
    'medium': {
        'task_description': 'document editing',
        'class': Medium
    },
    'bigcodebench': {
        'task_description': 'code generation',
        'class': BigCodeBench
    },
}