# Concurrency (goodput) sweep — `dflash` on `Qwen3.6-27B`

## Settings
- method (label): `dflash`
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
| `gsm8k` | 309.01 | 551.58 | 900.61 | 1,290.04 | 1,171.00 | 1,121.09 |
| `mbpp` | 275.81 | 475.84 | 744.37 | 1,036.25 | 1,053.82 | 1,050.05 |
| `mt-bench` | 168.01 | 305.87 | 514.84 | 763.45 | 638.25 | 637.98 |

## Mean acceptance length (spec_accept_length)
| dataset \ conc | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- | --- | --- |
| `gsm8k` | 7.411 | 7.420 | 7.392 | 7.387 | 7.400 | 7.397 |
| `mbpp` | 7.636 | 7.642 | 7.669 | 7.678 | 7.655 | 7.664 |
| `mt-bench` | 4.859 | 4.843 | 4.864 | 4.870 | 4.885 | 4.861 |

