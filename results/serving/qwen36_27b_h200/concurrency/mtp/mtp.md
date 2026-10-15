# Concurrency (goodput) sweep — `mtp` on `Qwen3.6-27B`

## Settings
- method (label): `mtp`
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
| `gsm8k` | 173.64 | 322.98 | 572.13 | 936.74 | 861.74 | 1,094.65 |
| `mbpp` | 163.49 | 294.60 | 498.11 | 774.50 | 745.45 | 915.76 |
| `mt-bench` | 146.47 | 271.58 | 483.54 | 793.92 | 744.95 | 926.18 |

## Mean acceptance length (spec_accept_length)
| dataset \ conc | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- | --- | --- |
| `gsm8k` | 3.676 | 3.676 | 3.671 | 3.674 | 3.675 | 3.677 |
| `mbpp` | 3.632 | 3.630 | 3.626 | 3.632 | 3.624 | 3.625 |
| `mt-bench` | 3.155 | 3.152 | 3.159 | 3.168 | 3.155 | 3.153 |

