# Conditioning ladder — raw data behind Table `tab:c3`

Three arms, separating the two things DominoTree changes at once relative to a
factorized (marginal) draft tree:

| arm | scorer | isolates |
|---|---|---|
| `marg@16` | base logits only — no GRU correction | — |
| `condstatic@16` | correction applied **once per depth** along the greedy path, then shared | having the correction **at all** |
| `dominotree@16` | correction recomputed **per realized path** | **path-dependence** (this paper's contribution) |

## Two builder configurations, both shipped here

| directory | builder | what it answers |
|---|---|---|
| `matched_builder/` | Python for all three | isolates the **scoring function** by holding construction cost fixed — the ablation proper |
| `best_builder/` | `marg` Python; `condstatic` and `dominotree` CUDA-graph captured | what a **deployment** sees with each arm at its fastest |

Each directory is a single collection session: all arms measured against one AR
measurement and paired per prompt, so machine-state drift is common-mode and cancels in
the ratio. (Comparing arms across sessions is not safe — we measured a 2.4 ms shift in
shared verify time between two sessions on the same box, which moved a ratio by ~3 points.)

## Headline

`τ` is **identical** across both configurations — 6.92 → 7.62 → 7.98 — because a builder
changes how the tree is found, never which nodes it contains. That is `+10.1%` for
applying the correction and `+4.7%` for conditioning on the realized path.

Throughput over the marginal tree is `+6.57%` [+5.48, +7.67] matched, `+12.56%`
[+11.41, +13.71] at best builders. The split moves between the two because graph capture
helps `condstatic` more than `dominotree`: a path-independent trajectory captures as one
graph, while best-first's data-dependent pop order forces one replay per node.

DominoTree carries the **highest** build cost of the three arms (2.36 ms vs 1.90 and
1.78), because per-node correction is strictly more work than per-depth correction. It
wins throughput anyway.

## Reproduce

```bash
python3 results/conditioning_ladder/ladder_ci.py results/conditioning_ladder/matched_builder
python3 results/conditioning_ladder/ladder_ci.py results/conditioning_ladder/best_builder
```

Per-directory `PROVENANCE.txt` records the exact protocol, builder configuration, and
measured per-round build costs.
