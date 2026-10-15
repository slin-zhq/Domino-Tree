# Concurrency (goodput) sweep — `ar` on `Qwen3.6-27B`

## Settings
- method (label): `ar`
- model: `Qwen3.6-27B`  (tokenizer/template: `/work/u8093362/models/Qwen3.6-27B`)
- host:port: `127.0.0.1:31300`
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
| `gsm8k` | 66.99 | 127.15 | 238.16 | 425.92 | 338.54 | 605.74 |
| `mbpp` | 66.44 | 124.55 | 227.47 | 384.15 | 311.00 | 514.32 |
| `mt-bench` | 67.17 | 127.33 | 235.56 | 419.89 | 335.40 | 582.78 |

## Mean acceptance length (spec_accept_length)
| dataset \ conc | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- | --- | --- |
| `gsm8k` | N/A | N/A | N/A | N/A | N/A | N/A |
| `mbpp` | N/A | N/A | N/A | N/A | N/A | N/A |
| `mt-bench` | N/A | N/A | N/A | N/A | N/A | N/A |

