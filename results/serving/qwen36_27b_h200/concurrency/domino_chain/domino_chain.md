# Concurrency (goodput) sweep — `domino_chain` on `Qwen3.6-27B`

## Settings
- method (label): `domino_chain`
- model: `Qwen3.6-27B`  (tokenizer/template: `/work/u8093362/models/Qwen3.6-27B`)
- host:port: `127.0.0.1:21292`
- max_new_tokens: `512`
- temperature/top_p/top_k: `0.0` / `1.0` / `-1`
- warmup_requests (batches of size=conc): `1`
- warmup_max_new_tokens: `256`
- skip_first: `0`
- concurrencies: `1, 2, 4, 8, 16, 32`
- prompt_selection: `shuffle(seed=0).select(range(N))` (== bs=1 prefix)
- warmup_pool: `post_timed_shuffled (disjoint from timed) or timed_reuse`
- metric: `tps = sum(completion_tokens)/wall_s ; mean_accept = mean(spec_accept_length)`

## Tasks
- `gsm8k`: N=`128`
- `mbpp`: N=`128`
- `mt-bench`: N=`80`

## Goodput (output tok/s)
| dataset \ conc | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- | --- | --- |
| `gsm8k` | 383.66 | 672.53 | 1,067.57 | 1,531.73 | 1,465.76 | 1,455.83 |
| `mbpp` | 280.80 | 479.86 | 757.04 | 1,082.91 | 1,078.01 | 1,090.62 |
| `mt-bench` | 183.51 | 334.31 | 558.85 | 833.21 | 730.83 | 726.69 |

## Mean acceptance length (spec_accept_length)
| dataset \ conc | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- | --- | --- |
| `gsm8k` | 9.321 | 9.326 | 9.321 | 9.313 | 9.309 | 9.293 |
| `mbpp` | 7.494 | 7.513 | 7.505 | 7.509 | 7.509 | 7.518 |
| `mt-bench` | 5.431 | 5.408 | 5.417 | 5.421 | 5.419 | 5.386 |

