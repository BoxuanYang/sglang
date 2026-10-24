#!/usr/bin/env python3
# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Run a FastServe-style ShareGPT RPS sweep against SGLang."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import statistics
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiohttp
import numpy as np

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RATES = tuple(range(1, 21))
DEFAULT_DURATION_PER_RATE_SECONDS = 300.0
SHAREGPT_REPO_ID = "anon8231489123/ShareGPT_Vicuna_unfiltered"
SHAREGPT_FILENAME = "ShareGPT_V3_unfiltered_cleaned_split.json"


@dataclass(frozen=True)
class RequestSpec:
    prompt: str
    prompt_tokens: int
    target_output_tokens: int


@dataclass(frozen=True)
class RequestMeasurement:
    request_index: int
    prompt_tokens: int
    target_output_tokens: int
    actual_output_tokens: int
    end_to_end_latency_seconds: float
    normalized_latency_seconds_per_token: float | None
    output_length_within_tolerance: bool | None
    success: bool
    error: str | None


@dataclass(frozen=True)
class RateGroupRun:
    measurements: list[RequestMeasurement]
    benchmark_duration_seconds: float
    submission_duration_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Poisson ShareGPT workload against SGLang and report the arithmetic "
            "mean of each request's end-to-end latency divided by its actual output tokens."
        )
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="/data/models/Qwen3-30B-A3B")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer name or path. Defaults to --model.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="ShareGPT JSON path. Downloads the standard dataset when omitted.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=SCRIPT_DIR / "results" / "summary.json",
    )
    parser.add_argument(
        "--request-output",
        type=Path,
        default=SCRIPT_DIR / "results" / "requests.json",
    )
    parser.add_argument(
        "--rates",
        type=float,
        nargs="+",
        default=list(DEFAULT_RATES),
        help="Offered request rates. Defaults to every integer from 1 through 20.",
    )
    parser.add_argument(
        "--duration-per-rate",
        type=float,
        default=DEFAULT_DURATION_PER_RATE_SECONDS,
        help=(
            "Target submission duration for each RPS group. num_prompts is "
            "ceil(RPS * duration) unless --num-prompts is supplied."
        ),
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Explicit request count: provide one value for every group or one value "
            "to use for all rates."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-total-tokens", type=int, default=8192)
    parser.add_argument("--output-token-tolerance", type=int, default=5)
    parser.add_argument("--tokenization-batch-size", type=int, default=512)
    parser.add_argument("--request-timeout", type=float, default=3 * 3600)
    parser.add_argument("--inter-group-delay", type=float, default=30.0)
    parser.add_argument(
        "--max-client-connections",
        type=int,
        default=0,
        help="aiohttp connection limit. Zero means unlimited.",
    )
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--disable-ignore-eos",
        action="store_true",
        help="Allow generation to stop at EOS before the ShareGPT target length.",
    )
    return parser.parse_args()


def resolve_num_prompts(args: argparse.Namespace) -> list[int]:
    if args.num_prompts is None:
        return [max(1, math.ceil(rate * args.duration_per_rate)) for rate in args.rates]
    if len(args.num_prompts) == 1:
        return args.num_prompts * len(args.rates)
    if len(args.num_prompts) != len(args.rates):
        raise ValueError(
            "--num-prompts must contain one value or exactly one value per --rates entry"
        )
    return args.num_prompts


def validate_args(args: argparse.Namespace, num_prompts: Sequence[int]) -> None:
    if not args.rates or any(
        rate <= 0 or not math.isfinite(rate) for rate in args.rates
    ):
        raise ValueError("Every request rate must be finite and positive")
    if args.duration_per_rate <= 0 or not math.isfinite(args.duration_per_rate):
        raise ValueError("--duration-per-rate must be finite and positive")
    if any(count <= 0 for count in num_prompts):
        raise ValueError("Every request count must be positive")
    if args.max_prompt_tokens <= 0 or args.max_total_tokens <= 0:
        raise ValueError("Token limits must be positive")
    if args.max_prompt_tokens > args.max_total_tokens:
        raise ValueError("--max-prompt-tokens cannot exceed --max-total-tokens")
    if args.output_token_tolerance < 0:
        raise ValueError("--output-token-tolerance cannot be negative")
    if args.tokenization_batch_size <= 0:
        raise ValueError("--tokenization-batch-size must be positive")
    if args.request_timeout <= 0:
        raise ValueError("--request-timeout must be positive")
    if args.inter_group_delay < 0:
        raise ValueError("--inter-group-delay cannot be negative")
    if args.max_client_connections < 0:
        raise ValueError("--max-client-connections cannot be negative")


def build_endpoint(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1/chat/completions"):
        return base_url
    if base_url.endswith("/v1"):
        return f"{base_url}/chat/completions"
    return f"{base_url}/v1/chat/completions"


def resolve_dataset(dataset: Path | None) -> Path:
    if dataset is not None:
        if not dataset.is_file():
            raise FileNotFoundError(f"ShareGPT dataset not found: {dataset}")
        return dataset

    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=SHAREGPT_REPO_ID,
            filename=SHAREGPT_FILENAME,
            repo_type="dataset",
        )
    )


def batched(values: Sequence[str], batch_size: int) -> Iterator[Sequence[str]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def token_lengths(
    tokenizer: PreTrainedTokenizerBase, values: Sequence[str], batch_size: int
) -> list[int]:
    lengths: list[int] = []
    for batch in batched(values, batch_size):
        encoded = tokenizer(
            list(batch),
            add_special_tokens=False,
            padding=False,
            truncation=False,
        ).input_ids
        lengths.extend(len(token_ids) for token_ids in encoded)
    return lengths


def load_eligible_requests(
    dataset_path: Path,
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_prompt_tokens: int,
    max_total_tokens: int,
    tokenization_batch_size: int,
) -> list[RequestSpec]:
    with dataset_path.open("r", encoding="utf-8") as file:
        raw_dataset = json.load(file)

    conversations = [
        item["conversations"]
        for item in raw_dataset
        if len(item.get("conversations", [])) >= 2
    ]
    prompts = [conversation[0]["value"] for conversation in conversations]
    completions = [conversation[1]["value"] for conversation in conversations]

    prompt_lengths = token_lengths(tokenizer, prompts, tokenization_batch_size)
    output_lengths = token_lengths(tokenizer, completions, tokenization_batch_size)

    eligible: list[RequestSpec] = []
    for prompt, prompt_len, output_len in zip(
        prompts, prompt_lengths, output_lengths, strict=True
    ):
        # FastServe rejects a too-short sample only when both sides are short.
        if prompt_len < 4 and output_len < 4:
            continue
        if prompt_len > max_prompt_tokens or prompt_len + output_len > max_total_tokens:
            continue
        eligible.append(
            RequestSpec(
                prompt=prompt,
                prompt_tokens=prompt_len,
                target_output_tokens=output_len,
            )
        )
    return eligible


def sample_requests(
    eligible: Sequence[RequestSpec], num_requests: int, seed: int
) -> list[RequestSpec]:
    if len(eligible) < num_requests:
        raise ValueError(
            f"Only {len(eligible)} requests pass the filters, but {num_requests} are required"
        )
    rng = random.Random(seed)
    sampled = rng.sample(list(eligible), num_requests)
    rng.shuffle(sampled)
    return sampled


async def send_request(
    *,
    session: aiohttp.ClientSession,
    request_index: int,
    request: RequestSpec,
    endpoint: str,
    model: str,
    headers: dict[str, str],
    ignore_eos: bool,
    output_token_tolerance: int,
) -> RequestMeasurement:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": request.prompt}],
        "n": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "max_completion_tokens": request.target_output_tokens,
        "ignore_eos": ignore_eos,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    actual_output_tokens = 0
    request_started = time.perf_counter()
    try:
        saw_done = False
        saw_usage = False
        async with session.post(endpoint, json=payload, headers=headers) as response:
            if response.status != 200:
                body = await response.text()
                raise RuntimeError(f"HTTP {response.status}: {body[:1000]}")

            async for raw_line in response.content:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if not data:
                    continue
                if data == "[DONE]":
                    saw_done = True
                    break

                event = json.loads(data)
                usage = event.get("usage") or {}
                if usage.get("completion_tokens") is not None:
                    actual_output_tokens = int(usage["completion_tokens"])
                    saw_usage = True

        request_finished = time.perf_counter()
        if not saw_done:
            raise RuntimeError("response stream ended before [DONE]")
        if not saw_usage:
            raise RuntimeError("response did not include usage.completion_tokens")
        if actual_output_tokens <= 0:
            raise RuntimeError("response generated zero completion tokens")

        latency = request_finished - request_started
        normalized_latency = latency / actual_output_tokens
        return RequestMeasurement(
            request_index=request_index,
            prompt_tokens=request.prompt_tokens,
            target_output_tokens=request.target_output_tokens,
            actual_output_tokens=actual_output_tokens,
            end_to_end_latency_seconds=latency,
            normalized_latency_seconds_per_token=normalized_latency,
            output_length_within_tolerance=(
                abs(actual_output_tokens - request.target_output_tokens)
                <= output_token_tolerance
            ),
            success=True,
            error=None,
        )
    except Exception as exc:
        request_finished = time.perf_counter()
        return RequestMeasurement(
            request_index=request_index,
            prompt_tokens=request.prompt_tokens,
            target_output_tokens=request.target_output_tokens,
            actual_output_tokens=actual_output_tokens,
            end_to_end_latency_seconds=request_finished - request_started,
            normalized_latency_seconds_per_token=None,
            output_length_within_tolerance=None,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
        )


async def run_rate_group(
    *,
    requests: Sequence[RequestSpec],
    request_rate: float,
    seed: int,
    endpoint: str,
    model: str,
    headers: dict[str, str],
    timeout: float,
    max_client_connections: int,
    ignore_eos: bool,
    output_token_tolerance: int,
) -> RateGroupRun:
    interval_rng = np.random.RandomState(seed)
    intervals = interval_rng.exponential(
        scale=1.0 / request_rate, size=max(0, len(requests) - 1)
    )

    completed = 0
    progress_step = max(1, len(requests) // 10)
    progress_lock = asyncio.Lock()

    async def measured(index: int, spec: RequestSpec) -> RequestMeasurement:
        nonlocal completed
        result = await send_request(
            session=session,
            request_index=index,
            request=spec,
            endpoint=endpoint,
            model=model,
            headers=headers,
            ignore_eos=ignore_eos,
            output_token_tolerance=output_token_tolerance,
        )
        async with progress_lock:
            completed += 1
            if completed % progress_step == 0 or completed == len(requests):
                print(
                    f"  RPS={request_rate:g}: {completed}/{len(requests)} requests completed",
                    flush=True,
                )
        return result

    client_timeout = aiohttp.ClientTimeout(total=timeout)
    connector = aiohttp.TCPConnector(limit=max_client_connections)
    group_started = time.perf_counter()
    async with aiohttp.ClientSession(
        timeout=client_timeout,
        connector=connector,
        read_bufsize=10 * 1024**2,
    ) as session:
        tasks: list[asyncio.Task[RequestMeasurement]] = []
        for index, spec in enumerate(requests):
            tasks.append(asyncio.create_task(measured(index, spec)))
            if index < len(intervals):
                await asyncio.sleep(float(intervals[index]))
        submission_finished = time.perf_counter()
        measurements = await asyncio.gather(*tasks)
    group_finished = time.perf_counter()

    return RateGroupRun(
        measurements=measurements,
        benchmark_duration_seconds=group_finished - group_started,
        submission_duration_seconds=submission_finished - group_started,
    )


def summarize_group(
    request_rate: float,
    num_prompts: int,
    group_run: RateGroupRun,
) -> dict[str, Any]:
    successful = [
        item
        for item in group_run.measurements
        if item.success and item.normalized_latency_seconds_per_token is not None
    ]
    normalized_latencies = [
        item.normalized_latency_seconds_per_token
        for item in successful
        if item.normalized_latency_seconds_per_token is not None
    ]
    total_input_tokens = sum(item.prompt_tokens for item in successful)
    total_output_tokens = sum(item.actual_output_tokens for item in successful)
    duration = group_run.benchmark_duration_seconds
    length_mismatches = sum(
        item.output_length_within_tolerance is False for item in successful
    )
    return {
        "request_rate": request_rate,
        "num_prompts": num_prompts,
        "benchmark_duration_seconds": duration,
        "submission_duration_seconds": group_run.submission_duration_seconds,
        "mean_normalized_latency_seconds_per_token": (
            statistics.fmean(normalized_latencies) if normalized_latencies else None
        ),
        "successful_requests": len(successful),
        "failed_requests": len(group_run.measurements) - len(successful),
        "output_length_mismatches": length_mismatches,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "achieved_request_throughput_requests_per_second": (
            len(successful) / duration if duration > 0 else None
        ),
        "input_token_throughput_tokens_per_second": (
            total_input_tokens / duration if duration > 0 else None
        ),
        "output_token_throughput_tokens_per_second": (
            total_output_tokens / duration if duration > 0 else None
        ),
    }


def build_metadata(
    args: argparse.Namespace,
    *,
    dataset_path: Path,
    tokenizer_name: str,
    endpoint: str,
    eligible_count: int,
    num_prompts: Sequence[int],
) -> dict[str, Any]:
    return {
        "benchmark": "FastServe-style ShareGPT RPS sweep against SGLang",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "dataset": str(dataset_path.resolve()),
        "model": args.model,
        "tokenizer": tokenizer_name,
        "rates": args.rates,
        "num_prompts": list(num_prompts),
        "duration_per_rate_seconds": args.duration_per_rate,
        "seed": args.seed,
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_total_tokens": args.max_total_tokens,
        "output_token_tolerance": args.output_token_tolerance,
        "ignore_eos": not args.disable_ignore_eos,
        "eligible_request_count": eligible_count,
        "latency_definition": "t_i = end_to_end_latency_i / actual_completion_tokens_i",
        "mean_definition": "arithmetic mean of t_i over successful requests",
        "arrival_process": "Poisson (exponential inter-arrival times)",
    }


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


async def run(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    num_prompts = resolve_num_prompts(args)
    validate_args(args, num_prompts)
    dataset_path = resolve_dataset(args.dataset)
    tokenizer_name = args.tokenizer or args.model
    endpoint = build_endpoint(args.base_url)

    print(f"Loading tokenizer from {tokenizer_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"Loading and tokenizing {dataset_path}...", flush=True)
    eligible = load_eligible_requests(
        dataset_path,
        tokenizer,
        max_prompt_tokens=args.max_prompt_tokens,
        max_total_tokens=args.max_total_tokens,
        tokenization_batch_size=args.tokenization_batch_size,
    )
    print(f"Eligible ShareGPT requests: {len(eligible)}", flush=True)

    metadata = build_metadata(
        args,
        dataset_path=dataset_path,
        tokenizer_name=tokenizer_name,
        endpoint=endpoint,
        eligible_count=len(eligible),
        num_prompts=num_prompts,
    )
    summary_document: dict[str, Any] = {"metadata": metadata, "results": []}
    request_document: dict[str, Any] = {"metadata": metadata, "groups": []}
    atomic_write_json(args.summary_output, summary_document)
    atomic_write_json(args.request_output, request_document)

    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    for group_index, (request_rate, request_count) in enumerate(
        zip(args.rates, num_prompts, strict=True), start=1
    ):
        print(
            f"[{group_index}/{len(args.rates)}] RPS={request_rate:g}, "
            f"num_prompts={request_count}",
            flush=True,
        )
        sampled = sample_requests(eligible, request_count, args.seed)
        group_run = await run_rate_group(
            requests=sampled,
            request_rate=request_rate,
            seed=args.seed,
            endpoint=endpoint,
            model=args.model,
            headers=headers,
            timeout=args.request_timeout,
            max_client_connections=args.max_client_connections,
            ignore_eos=not args.disable_ignore_eos,
            output_token_tolerance=args.output_token_tolerance,
        )

        summary = summarize_group(request_rate, request_count, group_run)
        summary_document["results"].append(summary)
        request_document["groups"].append(
            {
                "request_rate": request_rate,
                "num_prompts": request_count,
                "requests": [asdict(item) for item in group_run.measurements],
            }
        )
        atomic_write_json(args.summary_output, summary_document)
        atomic_write_json(args.request_output, request_document)

        mean = summary["mean_normalized_latency_seconds_per_token"]
        mean_text = "null" if mean is None else f"{mean * 1000:.3f} ms/token"
        request_throughput = summary["achieved_request_throughput_requests_per_second"]
        request_throughput_text = (
            "null" if request_throughput is None else f"{request_throughput:.3f} req/s"
        )
        output_throughput = summary["output_token_throughput_tokens_per_second"]
        output_throughput_text = (
            "null" if output_throughput is None else f"{output_throughput:.3f} tok/s"
        )
        print(
            f"  normalized_latency={mean_text}, "
            f"request_throughput={request_throughput_text}, "
            f"output_throughput={output_throughput_text}, "
            f"successful={summary['successful_requests']}, "
            f"failed={summary['failed_requests']}",
            flush=True,
        )

        if group_index < len(args.rates) and args.inter_group_delay:
            await asyncio.sleep(args.inter_group_delay)

    print(f"Wrote {args.summary_output}", flush=True)
    print(f"Wrote {args.request_output}", flush=True)


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
