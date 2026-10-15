# Concurrency (goodput) sweep — `dominotree@15` on `Qwen3.6-27B`

## Settings
- method (label): `dominotree@15`
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
| `gsm8k` | 389.99 | 673.72 | 1,057.88 | 1,468.02 | 1,373.40 | 1,378.10 |
| `mbpp` | 299.44 | 500.41 | 775.24 | 1,044.94 | 1,063.17 | 1,060.17 |
| `mt-bench` | 221.22 | 387.76 | 623.10 | 890.12 | 783.92 | 766.85 |

## Mean acceptance length (spec_accept_length)
| dataset \ conc | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- | --- | --- |
| `gsm8k` | 9.734 | 9.779 | 9.788 | 9.786 | 9.795 | 9.762 |
| `mbpp` | 8.043 | 8.090 | 8.076 | 8.082 | 8.068 | 8.090 |
| `mt-bench` | 6.140 | 6.150 | 6.150 | 6.159 | 6.127 | 6.186 |

