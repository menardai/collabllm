import json
import platform
import contextlib
import faulthandler
import io
import multiprocessing
import os
import signal
import tempfile
from typing import Any, Dict, List, Optional

from collabllm.metric import SingleTurnOrChatMetric, BaseMetric


@SingleTurnOrChatMetric.register_metric("pass_rate")
class BigCodeBenchPassRateMetric(BaseMetric):
    """
    Executes model-generated code against a unit-test suite.
    Assumes:
        • messages[0]['content'] – programming prompt
        • messages[-1]['content'] – JSON with keys: dataset, test, entry_point, …
    """

    # --------------- public API ---------------- #
    def score(
        self,
        prompt: str,
        groundtruth: str,
        completion: Optional[str],
        messages: Optional[List[Dict[str, str]]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if completion is None:
            raise ValueError("`completion` (candidate code) must be provided.")

        problem: Dict[str, Any] = {"prompt": messages[0]["content"]}
        problem.update(metadata)

        from bigcodebench.eval import untrusted_check
        res = untrusted_check(
            completion,
            problem["test"],
            problem["entry_point"],
            max_as_limit=300 * 1024,
            max_data_limit=300 * 1024,
            max_stack_limit=300 * 1024,
            min_time_limit=10,
            gt_time_limit=10,
        )
        passed, info = res[0] == "pass", res[1]

        return float(passed)


if __name__ == "__main__":
    # Example usage
    metric = PassRateMetric()
    messages = [
        {"role": "user", "content": "Write a function to add two numbers."},
        {"role": "assistant", "content": '{"dataset": "math", "test": "add_test", "entry_point": "add"}'}
    ]
    completion = "def add(a, b): return a + b"
    score = metric.score(
        prompt="",
        groundtruth="",
        completion=completion,
        messages=messages,
        metadata={"unit_tests": ["add_test"]},
    )
    print("Pass Rate Score:", score)