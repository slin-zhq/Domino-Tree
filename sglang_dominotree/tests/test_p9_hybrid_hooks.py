"""P9 hybrid-GDN hooks: CPU-only checks (no GPU, no model weights).

Covers the parts of P9 that can be wrong without a GPU: the hooks are scoped to
DOMINOTREE, the conv-window layout switch flips only there, the GDN backend init is
wrapped, and the hybrid-target detector fails closed. The end-to-end losslessness
check needs the target model and runs on GPU (nano5_p9_gate.sh).

    python -m pytest sglang_dominotree/tests/test_p9_hybrid_hooks.py -q
"""

import types

import pytest

pytest.importorskip("sglang")

import sglang.srt.mem_cache.memory_pool as mp  # noqa: E402
import sglang.srt.server_args as sa  # noqa: E402
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend  # noqa: E402

import dominotree_sglang  # noqa: E402
from dominotree_sglang.worker import _detect_hybrid_gdn_target  # noqa: E402


@pytest.fixture
def server_algo(monkeypatch):
    def set_algo(name):
        monkeypatch.setattr(
            sa, "get_global_server_args",
            lambda: types.SimpleNamespace(speculative_algorithm=name))
    return set_algo


def test_hooks_installed():
    dominotree_sglang.register_plugin()
    assert dominotree_sglang._HYBRID_TREE_HOOKS_INSTALLED
    assert getattr(mp.conv_window_dedup_enabled, "_dominotree_wrapped", False)
    assert getattr(GDNAttnBackend.__init__, "_dominotree_wrapped", False)


@pytest.mark.parametrize("algo,expect", [
    ("DOMINOTREE", False),   # tree verify: dense per-node conv windows
    ("DOMINO", True),        # chain: upstream default (dedup) untouched
    ("DFLASH", True),
    ("EAGLE", True),
    (None, True),            # AR
])
def test_conv_dedup_scoped_to_dominotree(server_algo, algo, expect):
    dominotree_sglang.register_plugin()
    server_algo(algo)
    assert mp.conv_window_dedup_enabled(False, False, 1) is expect


def test_conv_dedup_keeps_upstream_rules_elsewhere(server_algo):
    dominotree_sglang.register_plugin()
    server_algo("DOMINO")
    assert mp.conv_window_dedup_enabled(False, False, 4) is False  # EAGLE tree
    assert mp.conv_window_dedup_enabled(True, False, 1) is False   # NPU


def _mr(mambaish, gdn):
    return types.SimpleNamespace(mambaish_config=mambaish, hybrid_gdn_config=gdn)


def test_detector_attention_only_target():
    assert _detect_hybrid_gdn_target(_mr(None, None)) is False


def test_detector_hybrid_gdn_target():
    dominotree_sglang.register_plugin()
    cfg = object()
    assert _detect_hybrid_gdn_target(_mr(cfg, cfg)) is True


def test_detector_refuses_non_gdn_recurrent_target():
    with pytest.raises(NotImplementedError):
        _detect_hybrid_gdn_target(_mr(object(), None))


def test_detector_fails_closed_when_it_cannot_tell():
    with pytest.raises(RuntimeError):
        _detect_hybrid_gdn_target(types.SimpleNamespace())


def test_detector_refuses_without_hooks(monkeypatch):
    monkeypatch.setattr(dominotree_sglang, "_HYBRID_TREE_HOOKS_INSTALLED", False)
    monkeypatch.setattr(dominotree_sglang, "_HYBRID_DIAG_FLAT_GDN", False)
    cfg = object()
    with pytest.raises(RuntimeError):
        _detect_hybrid_gdn_target(_mr(cfg, cfg))
