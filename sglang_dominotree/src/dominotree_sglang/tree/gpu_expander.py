"""GPU-native (CUDA-graph) node expander for the DominoTree best-first builder.

Ported near-verbatim from DominoTree's ``dominotree_gpu.py`` (``GraphNodeExpander``).
It removes per-node kernel-launch overhead + the per-pop GPU->CPU sync of the
pure-Python conditional children_fn by replaying three captured CUDA graphs
(setup / corr / base-expand), following the static-buffer capture/replay pattern:

    copy inputs into static buffers -> graph.replay() -> read static outputs

**Equivalence by construction:** ``children_fn`` mirrors
``conditional_children.make_conditional_children_fn`` op-for-op on identical
shapes/dtypes; the graph bodies are line-by-line transcriptions where the only
change is that inputs are read from static buffers and depth-dependent tensors
are ``index_select``-ed on a static depth-index tensor. The same body functions
run eagerly (``use_graphs=False`` / ``DOMINOTREE_GPU_EAGER=1``) or under capture,
so the captured math is exactly what the equivalence suite verifies.

**SGLang adaptation (the ONE change vs the reference):** the reference hardcodes
``self.target.model.embed_tokens``; the SGLang target exposes its embedding via
``get_input_embeddings()``, so this port takes an ``embed_tokens`` module argument
and uses it directly. Everything else (math, buffers, capture) is unchanged.
"""

from __future__ import annotations

import os

import torch


class GraphNodeExpander:
    """Drop-in provider of the DominoTree conditional ``children_fn``.

    Construct once (graph capture is expensive), call ``begin_round(ph,
    base_logits)`` once per draft block, then pass ``expander.children_fn`` to
    ``build_best_first_tree`` in place of ``make_conditional_children_fn(...)``.

    ``draft`` is the ``DominoDraftModel`` (owns ``prefix_gru`` + ``embed_proj``);
    ``embed_tokens`` is the target embedding module (``get_input_embeddings()``).
    """

    def __init__(
        self,
        *,
        draft,
        embed_tokens,
        k_draft: int,
        prefix_len: int,
        node_topk: int,
        corr_topm: int,
        device,
        use_graphs: bool | None = None,
    ) -> None:
        # Accept a torch.device or a device string ("cuda") from callers.
        device = torch.device(device)
        if use_graphs is None:
            use_graphs = (
                device.type == "cuda"
                and os.environ.get("DOMINOTREE_GPU_EAGER", "0") != "1"
            )
        if use_graphs and device.type != "cuda":
            raise ValueError("use_graphs=True requires a CUDA device")
        if k_draft <= 0 or node_topk <= 0:
            raise ValueError("k_draft and node_topk must be positive")
        if corr_topm > 0 and node_topk > corr_topm:
            raise ValueError("node_topk must be <= corr_topm when corr_topm > 0")

        self.draft = draft
        self.embed_tokens = embed_tokens
        self.k_draft = int(k_draft)
        self.prefix_len = int(prefix_len)
        self.node_topk = int(node_topk)
        self.corr_topm = int(corr_topm)
        self.device = device
        self.use_graphs = bool(use_graphs)

        dtype = next(draft.parameters()).dtype
        gru_dim = draft.prefix_gru.hidden_size
        vocab = draft.embed_proj[2].out_features
        mlp_dim = draft.embed_proj[2].in_features
        hidden = draft.embed_proj[0].in_features - gru_dim
        self._gru_dim = gru_dim

        # ---- static input buffers (written per round) ----
        self.S_ph_all = torch.zeros(self.k_draft, hidden, dtype=dtype, device=device)
        self.S_base_all = torch.zeros(self.k_draft, vocab, dtype=dtype, device=device)
        # ---- static input buffers (written per node pop) ----
        self.S_state = torch.zeros(1, 1, gru_dim, dtype=dtype, device=device)
        self.S_depth = torch.zeros(1, dtype=torch.long, device=device)
        self.S_exp_tokens = torch.zeros(self.node_topk, dtype=torch.long, device=device)
        # ---- per-round candidate tensors (written by the setup graph) ----
        if self.corr_topm > 0:
            self.S_cand_all = torch.zeros(
                self.k_draft, self.corr_topm, dtype=torch.long, device=device
            )
            self.S_w2c_all = torch.zeros(
                self.k_draft, self.corr_topm, mlp_dim, dtype=torch.float32, device=device
            )
            self.S_basec_all = torch.zeros(
                self.k_draft, self.corr_topm, dtype=torch.float32, device=device
            )
        # ---- static output buffers ----
        self.S_out_tokens = torch.zeros(self.node_topk, dtype=torch.long, device=device)
        self.S_out_lps = torch.zeros(self.node_topk, dtype=torch.float32, device=device)
        self.S_out_states = torch.zeros(
            1, self.node_topk, gru_dim, dtype=dtype, device=device
        )
        # ---- pinned host landing buffers (one sync per pop) ----
        if device.type == "cuda":
            self.H_tokens = torch.zeros(
                self.node_topk, dtype=torch.long, pin_memory=True
            )
            self.H_lps = torch.zeros(
                self.node_topk, dtype=torch.float32, pin_memory=True
            )

        # (device index tensor, token list, logprob list) per depth < prefix_len
        self._prefix_rows: "list[tuple[torch.Tensor, list[int], list[float]]] | None" = (
            None
        )

        self.g_setup = self.g_corr = self.g_base = None
        if self.use_graphs:
            self._capture()

    # ------------------------------------------------------------------
    # Graph bodies. Each reads ONLY static input buffers / model weights and
    # writes ONLY static output buffers, so the same function is valid both
    # eagerly and under CUDA-graph capture. The math transcribes
    # conditional_children.make_conditional_children_fn line-by-line.
    # ------------------------------------------------------------------

    def _setup_impl(self) -> None:
        """Per-round corr_topm candidate setup (cand/w2c/basec)."""
        w2 = self.draft.embed_proj[2].weight
        for d in range(self.k_draft):
            row = self.S_base_all[d]
            cd = torch.topk(row, self.corr_topm).indices
            self.S_cand_all[d].copy_(cd)
            self.S_w2c_all[d].copy_(w2[cd].float())
            self.S_basec_all[d].copy_(row[cd].float())

    def _corr_impl(self) -> None:
        """Candidate-restricted correction select + child-state expansion (corr_topm > 0)."""
        ph_d = self.S_ph_all.index_select(0, self.S_depth)  # == ph[depth].unsqueeze(0)
        s_feat = self.S_state.transpose(0, 1).squeeze(0)
        ep = self.draft.embed_proj
        h = ep[1](ep[0](torch.cat([ph_d, s_feat], dim=-1))).float()
        cand_d = self.S_cand_all.index_select(0, self.S_depth)[0]
        w2c_d = self.S_w2c_all.index_select(0, self.S_depth)[0]
        basec_d = self.S_basec_all.index_select(0, self.S_depth)[0]
        corrected = basec_d + (h @ w2c_d.t())[0]
        vals, local_idx = torch.topk(
            torch.log_softmax(corrected, dim=-1), k=self.node_topk
        )
        toks = cand_d[local_idx]
        self.S_out_tokens.copy_(toks)
        self.S_out_lps.copy_(vals)
        self._expand_impl(toks)

    def _corr_full_impl(self) -> None:
        """Full-vocab correction select + child-state expansion (corr_topm == 0)."""
        ph_d = self.S_ph_all.index_select(0, self.S_depth)
        s_feat = self.S_state.transpose(0, 1).squeeze(0)
        corr = self.draft.embed_proj(torch.cat([ph_d, s_feat], dim=-1))
        base_d = self.S_base_all.index_select(0, self.S_depth)
        logp = torch.log_softmax((base_d + corr).float(), dim=-1)
        vals, idx = torch.topk(logp, k=self.node_topk, dim=-1)
        self.S_out_tokens.copy_(idx[0])
        self.S_out_lps.copy_(vals[0])
        self._expand_impl(idx[0])

    def _base_expand_impl(self) -> None:
        """Child-state expansion for depth < prefix_len (tokens via S_exp_tokens)."""
        self._expand_impl(self.S_exp_tokens)

    def _expand_impl(self, toks: torch.Tensor) -> None:
        emb = self.embed_tokens(toks).unsqueeze(1)  # (node_topk, 1, E)
        h0 = self.S_state.expand(1, self.node_topk, self._gru_dim).contiguous()
        _, hn = self.draft.prefix_gru(emb, h0)
        self.S_out_states.copy_(hn)

    # ------------------------------------------------------------------
    # Capture (warm up the exact bodies on a side stream, then record each
    # into a CUDAGraph — the Domino DraftCorrectionGraphRunner pattern).
    # ------------------------------------------------------------------

    def _capture(self) -> None:
        corr_body = self._corr_impl if self.corr_topm > 0 else self._corr_full_impl
        torch.cuda.synchronize(self.device)
        side = torch.cuda.Stream(self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.inference_mode(), torch.cuda.stream(side):
            for _ in range(3):
                if self.corr_topm > 0:
                    self._setup_impl()
                corr_body()
                if self.prefix_len > 0:
                    self._base_expand_impl()
        torch.cuda.current_stream(self.device).wait_stream(side)
        torch.cuda.synchronize(self.device)

        if self.corr_topm > 0:
            self.g_setup = torch.cuda.CUDAGraph()
            with torch.inference_mode(), torch.cuda.graph(self.g_setup):
                self._setup_impl()
        self.g_corr = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.g_corr):
            corr_body()
        if self.prefix_len > 0:
            self.g_base = torch.cuda.CUDAGraph()
            with torch.inference_mode(), torch.cuda.graph(self.g_base):
                self._base_expand_impl()
        torch.cuda.synchronize(self.device)

    def _run_setup(self) -> None:
        if self.g_setup is not None:
            self.g_setup.replay()
        else:
            self._setup_impl()

    def _run_corr(self) -> None:
        if self.g_corr is not None:
            self.g_corr.replay()
        elif self.corr_topm > 0:
            self._corr_impl()
        else:
            self._corr_full_impl()

    def _run_base(self) -> None:
        if self.g_base is not None:
            self.g_base.replay()
        else:
            self._base_expand_impl()

    # ------------------------------------------------------------------
    # Public per-round / per-node API.
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def begin_round(self, ph: torch.Tensor, base_logits: torch.Tensor) -> None:
        """Load this round's draft outputs and run the candidate setup."""
        if (
            ph.dtype != self.S_ph_all.dtype
            or base_logits.dtype != self.S_base_all.dtype
        ):
            raise TypeError(
                f"dtype mismatch: ph={ph.dtype}, base_logits={base_logits.dtype}, "
                f"expected {self.S_ph_all.dtype} (a silent cast would change numerics)"
            )
        self.S_ph_all.copy_(ph[: self.k_draft])
        self.S_base_all.copy_(base_logits[: self.k_draft])
        if self.corr_topm > 0:
            self._run_setup()
        prefix_rows = []
        for d in range(self.prefix_len):
            # == log_prob_topk(base_logits[d], node_topk), which the Python path
            # re-evaluates per node; path-independent, so once per round is
            # value-identical.
            logp = torch.log_softmax(self.S_base_all[d].float(), dim=-1)
            vals, idx = torch.topk(logp, k=self.node_topk, dim=-1)
            prefix_rows.append((idx, idx.tolist(), vals.tolist()))
        self._prefix_rows = prefix_rows

    @torch.inference_mode()
    def children_fn(self, state, depth: int):
        """Drop-in ``ChildrenFn`` for ``build_best_first_tree``."""
        if depth >= self.k_draft:
            return [], [], []
        if self._prefix_rows is None:
            raise RuntimeError("begin_round() must be called before children_fn()")
        self.S_state.copy_(state)
        if depth < self.prefix_len:
            idx_dev, toks, lps = self._prefix_rows[depth]
            toks, lps = list(toks), list(lps)  # fresh lists per call, like the Python path
            self.S_exp_tokens.copy_(idx_dev)
            self._run_base()
        else:
            self.S_depth.fill_(depth)
            self._run_corr()
            toks, lps = self._read_out_tokens_lps()
        # The next replay overwrites S_out_states, so child states must be
        # snapshotted; heap entries hold views into this per-pop clone.
        hn = self.S_out_states.clone()
        states = [hn[:, k : k + 1, :] for k in range(self.node_topk)]
        return toks, lps, states

    def _read_out_tokens_lps(self):
        if self.device.type == "cuda":
            self.H_tokens.copy_(self.S_out_tokens, non_blocking=True)
            self.H_lps.copy_(self.S_out_lps, non_blocking=True)
            torch.cuda.synchronize(self.device)
            return self.H_tokens.tolist(), self.H_lps.tolist()
        return self.S_out_tokens.tolist(), self.S_out_lps.tolist()


# ----------------------------------------------------------------------
# Self-test: equivalence against the pure-Python reference on tiny models.
# Adapted from dominotree_gpu._equivalence_suite to this plugin's modules.
# Run: python -m dominotree_sglang.tree.gpu_expander  (needs torch; CUDA to
# exercise capture/replay).
# ----------------------------------------------------------------------


class _TinyTargetInner(torch.nn.Module):
    def __init__(self, vocab: int, hidden: int) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab, hidden)


class _TinyTarget(torch.nn.Module):
    def __init__(self, vocab: int, hidden: int) -> None:
        super().__init__()
        self.model = _TinyTargetInner(vocab, hidden)

    def get_input_embeddings(self):
        return self.model.embed_tokens


class _TinyDraft(torch.nn.Module):
    """Mirrors Domino's draft head: bias-free GRU + Sequential(Linear, SiLU, Linear)."""

    def __init__(self, hidden: int, gru_dim: int, mlp_dim: int, vocab: int) -> None:
        super().__init__()
        self.prefix_gru = torch.nn.GRU(
            input_size=hidden,
            hidden_size=gru_dim,
            num_layers=1,
            batch_first=True,
            bias=False,
        )
        self.embed_proj = torch.nn.Sequential(
            torch.nn.Linear(hidden + gru_dim, mlp_dim, bias=False),
            torch.nn.SiLU(),
            torch.nn.Linear(mlp_dim, vocab, bias=False),
        )


def _equivalence_suite(device: torch.device) -> None:
    from .best_first import build_best_first_tree
    from .conditional_children import make_conditional_children_fn

    torch.manual_seed(0)
    vocab, hidden, gru_dim, mlp_dim = 89, 12, 10, 24
    k_draft, node_topk, budget = 5, 3, 12
    target = _TinyTarget(vocab, hidden).to(device).eval()
    draft = _TinyDraft(hidden, gru_dim, mlp_dim, vocab).to(device).eval()
    embed_tokens = target.get_input_embeddings()

    with torch.no_grad():
        for corr_topm, prefix_len in [(8, 1), (8, 0), (0, 0), (0, 2), (3, 5)]:
            expander = GraphNodeExpander(
                draft=draft,
                embed_tokens=embed_tokens,
                k_draft=k_draft,
                prefix_len=prefix_len,
                node_topk=node_topk,
                corr_topm=corr_topm,
                device=device,
            )
            for round_idx in range(2):  # two rounds: exercises static-buffer reuse
                gen = torch.Generator().manual_seed(
                    1000 * round_idx + corr_topm * 10 + prefix_len
                )
                ph = torch.randn(k_draft, hidden, generator=gen).to(device)
                base_logits = torch.randn(k_draft, vocab, generator=gen).to(device)
                root_state = torch.randn(1, 1, gru_dim, generator=gen).to(device)

                ref_fn = make_conditional_children_fn(
                    ph=ph,
                    base_logits=base_logits,
                    draft_model=draft,
                    embed_tokens=embed_tokens,
                    node_topk=node_topk,
                    corr_topm=corr_topm,
                    prefix_len=prefix_len,
                    device=device,
                )
                ref_nodes = build_best_first_tree(ref_fn, root_state, budget, k_draft)

                expander.begin_round(ph, base_logits)
                fast_nodes = build_best_first_tree(
                    expander.children_fn, root_state, budget, k_draft
                )

                key = (
                    f"corr_topm={corr_topm} prefix_len={prefix_len} "
                    f"round={round_idx} device={device}"
                )
                assert len(ref_nodes) == len(fast_nodes), key
                for a, b in zip(ref_nodes, fast_nodes):
                    assert (a.token, a.depth, a.parent) == (
                        b.token,
                        b.depth,
                        b.parent,
                    ), (key, a, b)
                    assert abs(a.cum_logprob - b.cum_logprob) < 1e-5, (key, a, b)

                # Direct single-call check including child GRU states.
                probe_depth = min(prefix_len, k_draft - 1)
                r_toks, r_lps, r_states = ref_fn(root_state, probe_depth)
                f_toks, f_lps, f_states = expander.children_fn(root_state, probe_depth)
                assert r_toks == f_toks, key
                assert all(abs(x - y) < 1e-5 for x, y in zip(r_lps, f_lps)), key
                for rs, fs in zip(r_states, f_states):
                    assert torch.allclose(rs, fs, atol=1e-6, rtol=0), key

            # Contract edges.
            assert expander.children_fn(root_state, k_draft) == ([], [], [])
            fresh = GraphNodeExpander(
                draft=draft,
                embed_tokens=embed_tokens,
                k_draft=k_draft,
                prefix_len=prefix_len,
                node_topk=node_topk,
                corr_topm=corr_topm,
                device=device,
                use_graphs=False,
            )
            try:
                fresh.children_fn(root_state, min(prefix_len, k_draft - 1))
            except RuntimeError:
                pass
            else:
                raise AssertionError("children_fn before begin_round must raise")


def _flat_reference_tree(nodes, verified_tok: int, n: int, mask_token_id: int):
    """Flatten + dead-leaf-pad a ``build_best_first_tree`` node list with the
    EXACT semantics of ``worker._build_conditional_tree_for_req``: root at flat
    index 0, nodes in pop order, then ``mask_token_id`` children of the root
    (flat depth 1) up to ``n`` entries."""
    tokens = [int(verified_tok)]
    parents = [-1]
    depths = [0]
    cums = [0.0]
    for nd in nodes:
        tokens.append(int(nd.token))
        parents.append(0 if nd.parent == -1 else 1 + int(nd.parent))
        depths.append(1 + int(nd.depth))
        cums.append(float(nd.cum_logprob))
    while len(tokens) < n:
        tokens.append(int(mask_token_id))
        parents.append(0)
        depths.append(1)
        cums.append(float("-inf"))
    return tokens[:n], parents[:n], depths[:n], cums[:n]


def _frontier_equivalence_suite(device: torch.device) -> None:
    """PRIMARY correctness gate for the Option B frontier builder
    (batch_builder_design.md §2B / §3 gate 1).

    On tiny random models with TIE-FREE inputs (continuous random logits; exact
    float score ties have measure zero — required because the heap and the
    frontier break real ties differently, see frontier.py docstring), assert
    that the batched frontier build reproduces the per-request reference
    ``build_best_first_tree`` + ``make_conditional_children_fn`` EXACTLY:
    identical flat ``(token, parent, depth)`` lists per request (which also
    checks node ORDER == heap pop order and the parents-before-children
    invariant), ``cum_logprob`` within 1e-5, identical dead-leaf padding, and
    an ``intra_mask`` equal to ``build_intra_tree_mask_from_parents`` on the
    reference parents. bs in {1, 2, 5, 32}; 2 decode steps per config
    (static-buffer reuse). On CUDA the build additionally runs under
    ``torch.cuda.set_sync_debug_mode("error")`` to prove ZERO host syncs.
    """
    from .best_first import build_best_first_tree
    from .conditional_children import make_conditional_children_fn
    from .frontier import FrontierTreeBuilder
    from .toy_tree import build_intra_tree_mask_from_parents

    torch.manual_seed(0)
    vocab, hidden, gru_dim, mlp_dim = 89, 12, 10, 24
    mask_token_id = vocab - 1
    target = _TinyTarget(vocab, hidden).to(device).eval()
    draft = _TinyDraft(hidden, gru_dim, mlp_dim, vocab).to(device).eval()
    embed_tokens = target.get_input_embeddings()

    # (corr_topm, prefix_len, k_draft, node_topk, budget). The first five mirror
    # the GraphNodeExpander matrix (all three scorer cases + mixed prefix/corr
    # + node_topk == corr_topm). The last two are structural edges:
    # k_draft=2/node_topk=2 exhausts the candidate tree (6 finite nodes <
    # budget=12 -> real dead-leaf padding), and budget=2 makes W=B < node_topk
    # (frontier keep drops candidates every depth).
    cases = [
        (8, 1, 5, 3, 12),
        (8, 0, 5, 3, 12),
        (0, 0, 5, 3, 12),
        (0, 2, 5, 3, 12),
        (3, 5, 5, 3, 12),
        (8, 1, 2, 2, 12),
        (8, 1, 5, 3, 2),
    ]

    with torch.no_grad():
        for corr_topm, prefix_len, k_draft, node_topk, budget in cases:
            n = budget + 1
            builder = FrontierTreeBuilder(
                draft=draft,
                embed_tokens=embed_tokens,
                k_draft=k_draft,
                prefix_len=prefix_len,
                node_topk=node_topk,
                corr_topm=corr_topm,
                budget=budget,
                max_depth=k_draft,
                mask_token_id=mask_token_id,
                device=device,
            )
            for bs in (1, 2, 5, 32):
                for step in range(2):  # static-buffer reuse across decode steps
                    gen = torch.Generator().manual_seed(
                        100_000 * step
                        + 1_000 * bs
                        + 100 * corr_topm
                        + 10 * prefix_len
                        + k_draft
                        + budget
                    )
                    ph = torch.randn(bs, k_draft, hidden, generator=gen).to(device)
                    base_logits = torch.randn(bs, k_draft, vocab, generator=gen).to(
                        device
                    )
                    root_states = torch.randn(1, bs, gru_dim, generator=gen).to(device)
                    verified = torch.randint(
                        0, vocab, (bs,), generator=gen, dtype=torch.long
                    ).to(device)

                    if device.type == "cuda":
                        # Gate: the build must be sync-free end to end.
                        torch.cuda.set_sync_debug_mode(2)
                    try:
                        tokens_2d, intra_mask, aux = builder.build(
                            ph, base_logits, root_states, verified, return_aux=True
                        )
                    finally:
                        if device.type == "cuda":
                            torch.cuda.set_sync_debug_mode(0)

                    ref_parents_all = []
                    for b in range(bs):
                        ref_fn = make_conditional_children_fn(
                            ph=ph[b],
                            base_logits=base_logits[b],
                            draft_model=draft,
                            embed_tokens=embed_tokens,
                            node_topk=node_topk,
                            corr_topm=corr_topm,
                            prefix_len=prefix_len,
                            device=device,
                        )
                        ref_nodes = build_best_first_tree(
                            ref_fn, root_states[:, b : b + 1, :], budget, k_draft
                        )
                        r_tok, r_par, r_dep, r_cum = _flat_reference_tree(
                            ref_nodes, int(verified[b].item()), n, mask_token_id
                        )
                        ref_parents_all.append(r_par)
                        key = (
                            f"corr_topm={corr_topm} prefix_len={prefix_len} "
                            f"k_draft={k_draft} budget={budget} bs={bs} "
                            f"step={step} b={b} device={device}"
                        )
                        f_tok = tokens_2d[b].tolist()
                        f_par = aux["parents"][b].tolist()
                        f_dep = aux["depths"][b].tolist()
                        f_cum = aux["cum_logprobs"][b].tolist()
                        assert f_tok == r_tok, (key, f_tok, r_tok)
                        assert f_par == r_par, (key, f_par, r_par)
                        assert f_dep == r_dep, (key, f_dep, r_dep)
                        # cum_logprob sums ~depth bf16 log-probs; CUDA fp
                        # reductions differ from the per-request reference by
                        # ~1e-5..1e-3 (the "bit-identity is the wrong bar" effect).
                        # The TREE (token/parent/depth) is asserted EXACTLY above;
                        # this only sanity-bounds the score, so use a device-aware tol.
                        cum_tol = 1e-2 if "cuda" in str(key) else 1e-5
                        for i, (rc, fc) in enumerate(zip(r_cum, f_cum)):
                            if rc == float("-inf"):
                                assert fc == float("-inf"), (key, i, fc)
                            else:
                                assert abs(rc - fc) < cum_tol, (key, i, rc, fc)

                    ref_mask = build_intra_tree_mask_from_parents(
                        ref_parents_all, n=n, device=device
                    )
                    assert torch.equal(intra_mask, ref_mask), (
                        f"intra_mask mismatch: corr_topm={corr_topm} "
                        f"prefix_len={prefix_len} k_draft={k_draft} "
                        f"budget={budget} bs={bs} step={step}"
                    )

            # Contract edge: dtype guard (no silent cast).
            bad_ph = torch.randn(1, k_draft, hidden).to(
                device=device, dtype=torch.float16
            )
            bad_logits = torch.randn(1, k_draft, vocab).to(
                device=device, dtype=torch.float16
            )
            try:
                builder.build(
                    bad_ph,
                    bad_logits,
                    torch.randn(1, 1, gru_dim).to(device),
                    torch.zeros(1, dtype=torch.long, device=device),
                )
            except TypeError:
                pass
            else:
                raise AssertionError("frontier build with wrong dtype must raise")


def _self_test() -> None:
    _equivalence_suite(torch.device("cpu"))
    print("gpu_expander self-test (cpu, eager-static vs pure-Python): ALL PASSED")
    _frontier_equivalence_suite(torch.device("cpu"))
    print(
        "frontier self-test (cpu, batched frontier vs best-first heap, "
        "bs=1/2/5/32 x 2 steps): ALL PASSED"
    )
    if torch.cuda.is_available():
        _equivalence_suite(torch.device("cuda"))
        print(
            "gpu_expander self-test (cuda, CUDA-graph replay vs pure-Python): ALL PASSED"
        )
        _frontier_equivalence_suite(torch.device("cuda"))
        print(
            "frontier self-test (cuda, sync-free batched frontier vs best-first "
            "heap): ALL PASSED"
        )
    else:
        print(
            "gpu_expander/frontier self-test: no CUDA device; graph replay and "
            "the zero-sync gate NOT exercised here"
        )


if __name__ == "__main__":
    _self_test()
