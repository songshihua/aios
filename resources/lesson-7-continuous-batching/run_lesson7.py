"""
Lesson 7 runner: continuous batching demo.

Submits more requests than the running-set cap; the scheduler admits up to
`max_running` requests, immediately backfilling slots as short requests finish.

Usage:
    python run_lesson7.py --model Qwen/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import time
from random import randint, seed
from typing import List

from aios.core import SamplingParams
from aios.llm import LLM


def _build_workload(num_seqs: int, prompt_len: int, out_low: int, out_high: int):
    prompts: List[List[int]] = [
        [randint(0, 10000) for _ in range(prompt_len)] for _ in range(num_seqs)
    ]
    params = [
        SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=randint(out_low, out_high),
        )
        for _ in range(num_seqs)
    ]
    return prompts, params


def main():
    parser = argparse.ArgumentParser(description="Lesson 7 continuous-batching demo")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--num-seqs", type=int, default=64)
    parser.add_argument("--prompt-len", type=int, default=256)
    parser.add_argument("--out-low", type=int, default=32)
    parser.add_argument("--out-high", type=int, default=256)
    parser.add_argument("--max-running", type=int, default=32, help="Running-set cap")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed(args.seed)
    prompts, params = _build_workload(
        args.num_seqs, args.prompt_len, args.out_low, args.out_high
    )
    total_tokens = sum(sp.max_tokens for sp in params)

    llm = LLM(args.model)

    # Warm-up: 1 short greedy request to prime kernels.
    llm.generate(
        [[randint(0, 10000) for _ in range(16)]],
        SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=4),
    )

    print(
        f"\nWorkload: num_seqs={args.num_seqs}, prompt_len={args.prompt_len}, "
        f"out={args.out_low}..{args.out_high}, total_tokens={total_tokens}, "
        f"max_running={args.max_running}"
    )

    t0 = time.time()
    llm.generate(prompts, params, max_running_reqs=args.max_running)
    t = time.time() - t0
    print(
        f"[CONTINUOUS] max_running={args.max_running} "
        f"elapsed={t:.2f}s, throughput={total_tokens / t:.2f} tok/s"
    )


if __name__ == "__main__":
    main()
