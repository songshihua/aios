"""Lesson 11 runner: per-request sampling versus vectorized batch sampling.

Usage:
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python \
    python resources/lesson-11-sampling/run_lesson11.py \
      --model /data4/home/yan.wang/huggingface/Qwen3-0.6B \
      --suite all --cuda-graph
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from types import SimpleNamespace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lesson 11 sampling runner")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--cuda-visible-devices", type=str, default=None)
    parser.add_argument(
        "--suite", choices=("check", "bench", "e2e", "all"), default="all"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--max-running", type=int, default=4)
    parser.add_argument("--memory-ratio", type=float, default=0.2)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def configure_env(args: argparse.Namespace) -> None:
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices


def make_batch(params):
    reqs = [SimpleNamespace(sampling_params=param) for param in params]
    return SimpleNamespace(reqs=reqs)


@dataclass
class LegacySampler:
    """Lesson 10 baseline: one Python sampling call per request."""

    sampling_params: object

    def sample(self, logits):
        import torch
        import torch.nn.functional as F

        params = self.sampling_params
        if params.is_greedy:
            return logits.argmax(dim=-1, keepdim=True)

        logits = logits / params.temperature
        if params.top_k > 0:
            topk_values = torch.topk(
                logits, min(params.top_k, logits.size(-1))
            ).values
            logits = logits.masked_fill(
                logits < topk_values[..., -1:], float("-inf")
            )
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)


def run_checks(args: argparse.Namespace) -> None:
    import torch

    from aios.core import SamplingParams
    from aios.engine.sample import Sampler

    device = torch.device("cuda:0")
    sampler = Sampler(device, args.vocab_size)

    greedy_params = [SamplingParams(temperature=0.0) for _ in range(4)]
    greedy_logits = torch.tensor(
        [[0.0, 3.0, 1.0], [4.0, 0.0, 1.0], [0.0, 2.0, 5.0], [1.0, 0.0, 0.0]],
        device=device,
    )
    greedy_args = sampler.prepare(make_batch(greedy_params))
    greedy_tokens = sampler.sample(greedy_logits, greedy_args)
    assert greedy_args.temperatures is None
    assert greedy_tokens.tolist() == [1, 0, 2, 0]
    print("[CHECK] all-greedy fast path: PASS")

    mixed_params = [
        SamplingParams(temperature=0.0),
        SamplingParams(temperature=0.7),
        SamplingParams(temperature=0.8, top_k=8),
        SamplingParams(temperature=0.9, top_p=0.85),
        SamplingParams(temperature=1.0, top_k=12, top_p=0.9),
    ]
    mixed_args = sampler.prepare(make_batch(mixed_params))
    assert mixed_args.temperatures is not None
    assert mixed_args.top_k is not None
    assert mixed_args.top_p is not None
    assert mixed_args.temperatures.shape == (len(mixed_params),)
    assert mixed_args.top_k.tolist() == [args.vocab_size, args.vocab_size, 8, args.vocab_size, 12]
    torch.manual_seed(args.seed)
    mixed_logits = torch.randn(
        len(mixed_params), args.vocab_size, device=device, dtype=torch.float16
    )
    mixed_tokens = sampler.sample(mixed_logits, mixed_args)
    assert mixed_tokens.shape == (len(mixed_params),)
    assert mixed_tokens.dtype == torch.int32
    assert bool(((mixed_tokens >= 0) & (mixed_tokens < args.vocab_size)).all())
    print("[CHECK] mixed per-request arguments: PASS")

    support_batch_size = 256
    support_logits = torch.full(
        (support_batch_size, 16), -20.0, device=device, dtype=torch.float32
    )
    support_logits[:, :3] = torch.tensor([3.0, 2.0, 1.0], device=device)
    support_sampler = Sampler(device, 16)
    support_params = [
        SamplingParams(temperature=1.0, top_k=2)
        for _ in range(support_batch_size)
    ]
    support_tokens = support_sampler.sample(
        support_logits, support_sampler.prepare(make_batch(support_params))
    )
    assert bool((support_tokens < 2).all())
    print("[CHECK] top-k support constraint: PASS")

    nucleus_params = [
        SamplingParams(temperature=1.0, top_p=0.5)
        for _ in range(support_batch_size)
    ]
    nucleus_tokens = support_sampler.sample(
        support_logits, support_sampler.prepare(make_batch(nucleus_params))
    )
    assert bool((nucleus_tokens == 0).all())
    print("[CHECK] top-p support constraint: PASS")


def benchmark_call(fn, iterations: int) -> float:
    import torch

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iterations


def run_bench(args: argparse.Namespace) -> None:
    import torch

    from aios.core import SamplingParams
    from aios.engine.sample import Sampler

    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    logits = torch.randn(
        args.batch_size,
        args.vocab_size,
        device=device,
        dtype=torch.float16,
    )
    params = [SamplingParams(temperature=0.8) for _ in range(args.batch_size)]
    sampler = Sampler(device, args.vocab_size)
    sampling_args = sampler.prepare(make_batch(params))

    def legacy_sample():
        return torch.cat(
            [
                LegacySampler(param).sample(logits[i : i + 1])
                for i, param in enumerate(params)
            ]
        )

    def batch_sample():
        return sampler.sample(logits, sampling_args)

    legacy_seconds = benchmark_call(legacy_sample, args.iterations)
    batch_seconds = benchmark_call(batch_sample, args.iterations)
    legacy_tps = args.batch_size / legacy_seconds
    batch_tps = args.batch_size / batch_seconds
    print(
        f"[LEGACY_PER_REQ] batch={args.batch_size} vocab={args.vocab_size} "
        f"latency={legacy_seconds * 1e3:.3f}ms tokens_per_second={legacy_tps:.2f}"
    )
    print(
        f"[BATCH_FLASHINFER] batch={args.batch_size} vocab={args.vocab_size} "
        f"latency={batch_seconds * 1e3:.3f}ms tokens_per_second={batch_tps:.2f}"
    )
    print(f"[SPEEDUP] {legacy_seconds / batch_seconds:.2f}x")


def run_e2e(args: argparse.Namespace) -> None:
    from aios import LLM, SamplingParams

    prompts = [
        [151644, 872, 198],
        [151644, 872, 198],
        [151644, 8948, 198],
        [151644, 3838, 198],
    ]
    params = [
        SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=args.max_tokens),
        SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=args.max_tokens),
        SamplingParams(
            temperature=0.8,
            top_k=20,
            ignore_eos=True,
            max_tokens=args.max_tokens,
        ),
        SamplingParams(
            temperature=0.9,
            top_k=30,
            top_p=0.9,
            ignore_eos=True,
            max_tokens=args.max_tokens,
        ),
    ]
    llm = LLM(
        args.model,
        max_running_reqs=args.max_running,
        memory_ratio=args.memory_ratio,
        enable_cuda_graph=args.cuda_graph,
        cuda_graph_bs=[1, 2, 4] if args.cuda_graph else None,
    )
    try:
        results = llm.generate(
            prompts,
            params,
            max_running_reqs=min(args.max_running, len(prompts)),
        )
        assert len(results) == len(prompts)
        assert all(len(result["token_ids"]) == args.max_tokens for result in results)
        assert results[0]["token_ids"] == results[1]["token_ids"]
        print(
            f"[E2E] mixed sampling with cuda_graph={args.cuda_graph}: PASS "
            f"lengths={[len(result['token_ids']) for result in results]}"
        )
        for index, result in enumerate(results):
            print(f"  req={index} token_ids={result['token_ids']}")
    finally:
        llm.close()


def main() -> None:
    args = parse_args()
    configure_env(args)
    if args.suite in ("check", "all"):
        run_checks(args)
    if args.suite in ("bench", "all"):
        run_bench(args)
    if args.suite in ("e2e", "all"):
        run_e2e(args)


if __name__ == "__main__":
    main()
