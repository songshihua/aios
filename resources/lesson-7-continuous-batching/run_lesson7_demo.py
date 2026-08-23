"""
Lesson 7 demo workload — matches continuous_scheduling.excalidraw exactly.

4 requests (A, B, C, D) with output lengths (2, 5, 3, 2) and max_running=2.
Greedy + ignore_eos so the iter-by-iter trace is fully deterministic and
matches the diagram. Useful for stepping through the scheduler in a debugger.

Expected 8-iter pattern (see the diagram for the swim-lane view):
    iter 0  PREFILL A             pending=[B,C,D]   running=[A]
    iter 1  PREFILL B             pending=[C,D]     running=[A,B]   cap reached
    iter 2  DECODE  {A,B}  A✓     pending=[C,D]     running=[B]     free(A)
    iter 3  PREFILL C             pending=[D]       running=[B,C]
    iter 4  DECODE  {B,C}         pending=[D]       running=[B,C]
    iter 5  DECODE  {B,C}  C✓     pending=[D]       running=[B]     free(C)
    iter 6  PREFILL D             pending=[]        running=[B,D]
    iter 7  DECODE  {B,D}  B✓ D✓  pending=[]        running=[]      free(B,D)

Suggested breakpoints:
    python/aios/scheduler/scheduler.py  : schedule_next_batch, process_batch_output
    python/aios/scheduler/prefill.py    : _can_admit, schedule_next_batch
    python/aios/scheduler/decode.py     : filter_reqs, schedule_next_batch
"""

from __future__ import annotations

import argparse
from random import randint, seed
from typing import List

from aios.core import SamplingParams
from aios.llm import LLM


# Exact workload from the diagram.
OUT_LENS: List[int] = [2, 5, 3, 2]
LABELS: List[str] = ["A", "B", "C", "D"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lesson 7 continuous-batching debug workload (matches the diagram)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/data4/home/yan.wang/huggingface/Qwen3-0.6B",
        help="Path to model directory or HuggingFace model name",
    )
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=8,
        help="Tokens per prompt (all 4 prompts share this length, keeps trace short)",
    )
    parser.add_argument(
        "--max-running",
        type=int,
        default=2,
        help="Concurrency cap — keep at 2 to match the diagram",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    seed(args.seed)

    prompts: List[List[int]] = [
        [randint(0, 10000) for _ in range(args.prompt_len)] for _ in OUT_LENS
    ]
    params = [
        SamplingParams(
            temperature=0.0,    # greedy → deterministic stepping
            ignore_eos=True,    # force exact out_len so the trace matches the diagram
            max_tokens=ol,
        )
        for ol in OUT_LENS
    ]

    print(
        f"\nWorkload: 4 reqs · prompt_len={args.prompt_len} · "
        f"out_len={dict(zip(LABELS, OUT_LENS))} · max_running={args.max_running}\n"
    )

    llm = LLM(args.model, device=args.device)
    results = llm.generate(prompts, params, max_running_reqs=args.max_running)

    for label, r in zip(LABELS, results):
        print(
            f"req {label} (uid={r['uid']}): "
            f"generated {len(r['token_ids'])} tokens "
            f"(expected {OUT_LENS[r['uid']]})"
        )


if __name__ == "__main__":
    main()
