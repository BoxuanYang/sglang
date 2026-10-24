# FastServe-style ShareGPT benchmark for SGLang

This benchmark replays ShareGPT requests with Poisson arrivals against SGLang's
OpenAI-compatible Chat Completions endpoint. It sweeps RPS 1 through 20 by default.

For request `i`, it measures end-to-end latency `T_i` from HTTP submission until the
stream's `[DONE]` event. This includes server queueing, prefill, and decode. SGLang
reports the actual completion-token count `N_i` in the final usage chunk. The metric
for one request is:

```text
t_i = T_i / N_i
```

The reported normalized latency is the arithmetic mean of `t_i` over successful
requests in one RPS group. Failed requests are listed separately and excluded because
`t_i` is undefined when no completion-token count is available.

## Run

Start SGLang:

```bash
CUDA_VISIBLE_DEVICES=3 sglang serve /data/models/Qwen3-30B-A3B \
  --host 0.0.0.0 \
  --port 30000
```

Run the benchmark from the SGLang repository root:

```bash
python benchmark/fastserve/bench_sharegpt_normalized_latency.py \
  --base-url http://127.0.0.1:30000 \
  --model /data/models/Qwen3-30B-A3B \
  --tokenizer /data/models/Qwen3-30B-A3B
```

When `--dataset` is omitted, the script downloads the standard
`anon8231489123/ShareGPT_Vicuna_unfiltered` dataset through Hugging Face. To use a
local copy, add:

```bash
--dataset /path/to/ShareGPT_V3_unfiltered_cleaned_split.json
```

The default request count is `ceil(RPS * 300)` for each group, giving 300 requests at
RPS 1 and 6000 requests at RPS 20. This keeps the offered-load window at five minutes
per group. The full default sweep submits 63,000 requests and takes at least 100
minutes, plus queue drain and the inter-group delays.

For a shorter one-minute sweep:

```bash
python benchmark/fastserve/bench_sharegpt_normalized_latency.py \
  --duration-per-rate 60
```

To use one fixed request count at every RPS:

```bash
python benchmark/fastserve/bench_sharegpt_normalized_latency.py \
  --num-prompts 1000
```

You can also provide one count per rate. For example:

```bash
python benchmark/fastserve/bench_sharegpt_normalized_latency.py \
  --rates 1 5 10 15 20 \
  --num-prompts 300 1500 3000 4500 6000
```

The script writes group summaries to `benchmark/fastserve/results/summary.json` and
per-request measurements to `benchmark/fastserve/results/requests.json`. Both files
are updated after every completed RPS group so interrupted sweeps retain finished
results.

For high-backlog tests, raise the client machine's file-descriptor limit before
running. `--max-client-connections 0` is the default and leaves the aiohttp connection
pool unlimited so the client does not impose an extra concurrency cap.
