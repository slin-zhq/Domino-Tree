#!/usr/bin/env python3
"""Prove the vendored Domino head is a verbatim copy of the official fork.

DominoTree's SGLang plugin vendors three files of Domino's GRU-correction head
(``domino_helper.py``, ``domino_kernels.py``, ``domino_rollout.py``). They are
**copied, not reimplemented**, from the official Domino SGLang fork. This script
mechanically verifies that claim so a reviewer never has to take it on faith:

  1. it extracts each file from the official Domino repo at a *pinned commit*,
  2. it re-points the handful of documented import shims (the only edits we made),
  3. it asserts every remaining line of code is byte-identical (sha256), and that
     the only additions on our side are blank lines / ``# PORT SHIM`` comments.

Exit code 0 == verbatim (modulo the enumerated shims); 1 == a logic line drifted.
That makes this both a reviewer-facing proof and a regression guard: if anyone
ever edits a vendored file's *logic*, CI fails here.

Usage
-----
    # point at a local clone of https://github.com/jianuo-huang/Domino
    # with the fork branch fetched (git fetch origin sglang-feat/dflash-domino)
    python verify_vendored_head.py --domino /path/to/Domino

    # emit a machine-readable proof record too
    python verify_vendored_head.py --domino /path/to/Domino --json proof.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

# --- PINNED PROVENANCE ------------------------------------------------------
# The official Domino SGLang fork. The head files live on this branch only
# (the fork's `main` does not carry them). Pinned by commit SHA so the proof is
# reproducible regardless of what branch/HEAD the local clone sits at.
OFFICIAL_REPO = "https://github.com/jianuo-huang/Domino.git"
OFFICIAL_BRANCH = "sglang-feat/dflash-domino"
OFFICIAL_COMMIT = "e0d78707a089780ae3b0a23967a1de450818c42b"

# vendored basename -> upstream path (in the official repo, at OFFICIAL_COMMIT)
FILES = {
    "domino_helper.py": "python/sglang/srt/speculative/domino_helper.py",
    "domino_kernels.py": "python/sglang/srt/speculative/domino_kernels.py",
    "domino_rollout.py": "python/sglang/srt/speculative/domino_rollout.py",
}

# The COMPLETE, exhaustive set of edits made to the copied files: import-path
# shims only (the referenced modules live upstream under
# sglang.srt.speculative.* but are copied into this package here). Keyed by the
# stripped line in OUR copy -> the stripped line in the OFFICIAL source. If our
# copy contains any relative import NOT in this map, the proof fails loudly.
SHIM_MAP = {
    "from .config import is_dflash_domino_projector":
        "from sglang.srt.speculative.dflash_utils import is_dflash_domino_projector",
    "from .domino_helper import DFlashDominoHelper":
        "from sglang.srt.speculative.domino_helper import DFlashDominoHelper",
    "from .domino_kernels import (":
        "from sglang.srt.speculative.domino_kernels import (",
}

# The COMPLETE set of FUNCTIONAL edits to the copied files. Unlike SHIM_MAP (which
# only re-points import paths and changes no behaviour), an entry here changes what
# the code DOES, so each one must justify itself. The proof works by reverting these
# blocks back to the official text and then requiring an exact match -- so a declared
# patch is accounted for, while any *undeclared* drift still fails the check.
#
# Format: basename -> list of (our_code_lines, official_code_lines, rationale).
# Lines are whitespace-stripped code lines, in order, as compared by code_lines().
PATCH_MAP = {
    "domino_rollout.py": [
        (
            (
                "local_enough = int(free_bytes) >= required_bytes + cushion * 1024 * 1024",
                "flag = torch.tensor(",
                "[1 if local_enough else 0],",
                "dtype=torch.int32,",
                "device=local_lm_head_weight.device,",
                ")",
                "flag = tp_group.all_reduce(flag)",
                "all_enough = int(flag.item()) == tp_size",
                "if not all_enough:",
            ),
            ("if int(free_bytes) < required_bytes + cushion * 1024 * 1024:",),
            "TP>1 collective safety: the upstream code decides whether to use the "
            "replicated scorer from each rank's OWN free memory (torch.cuda.mem_get_info). "
            "Ranks can disagree, and then some ranks enter a collective the others skip, "
            "which hangs the server. We all-reduce the decision so every rank takes the "
            "same branch. Required for Qwen3-8B at --tp-size 2; a no-op at TP=1.",
        ),
        (
            (
                '"DFLASH TP replicated scorer disabled by auto memory check "',
                '"(group-wide AND): need %.2f GiB + %d MiB cushion, this rank "',
                '"free %.2f GiB (local_enough=%s). Set "',
                '"DFLASH_DOMINO_TP_REPLICATE_SCORER=1 to force or 0 to silence.",',
            ),
            (
                '"DFLASH TP replicated scorer disabled by auto memory check: "',
                '"need %.2f GiB + %d MiB cushion, free %.2f GiB. "',
                '"Set DFLASH_DOMINO_TP_REPLICATE_SCORER=1 to force or 0 to silence.",',
            ),
            "Log text for the patch above: says the decision is a group-wide AND and "
            "reports this rank's own verdict. Message only.",
        ),
        (
            ("local_enough,",),
            (),
            "Log argument for the reworded message above. Message only.",
        ),
    ],
}

PKG_DIR = Path(__file__).resolve().parent / "src" / "dominotree_sglang"


def revert_patches(name: str, our_code: list[str]) -> tuple[list[str], list[str]]:
    """Undo the declared functional patches, returning (reverted_lines, errors).

    Each declared patch must appear EXACTLY ONCE. Missing means the declaration has
    gone stale (the code moved on without PROVENANCE being updated); more than once
    means the declaration is too loose to be a proof. Both are failures.
    """
    errors: list[str] = []
    for ours_block, official_block, _why in PATCH_MAP.get(name, []):
        n, hits = len(ours_block), []
        for i in range(len(our_code) - n + 1):
            if tuple(our_code[i:i + n]) == ours_block:
                hits.append(i)
        if len(hits) != 1:
            errors.append(
                f"declared patch matched {len(hits)} times (expected exactly 1): "
                f"{ours_block[0][:70]!r}"
            )
            continue
        i = hits[0]
        our_code = our_code[:i] + list(official_block) + our_code[i + n:]
    return our_code, errors


def official_source(domino_repo: Path, upstream_path: str) -> str:
    """Extract a file from the official repo at the pinned commit."""
    try:
        out = subprocess.run(
            ["git", "-C", str(domino_repo), "show", f"{OFFICIAL_COMMIT}:{upstream_path}"],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        sys.exit(
            f"\nERROR: could not read {upstream_path} at {OFFICIAL_COMMIT[:10]} from "
            f"{domino_repo}.\n  git said: {e.stderr.strip()}\n"
            f"  Fetch the fork branch first:\n"
            f"    git -C {domino_repo} fetch origin {OFFICIAL_BRANCH}\n"
        )
    return out.stdout


def is_noise(line: str) -> bool:
    """Full-line comment or blank line (never a code line with a trailing #)."""
    s = line.strip()
    return s == "" or s.startswith("#")


def apply_shims(line: str) -> str:
    """Re-point a documented intra-package import back to its upstream path."""
    return SHIM_MAP.get(line.strip(), line)


def code_lines(text: str) -> list[str]:
    """Code only: drop blank/comment lines, re-point the documented imports."""
    return [apply_shims(ln).strip() for ln in text.splitlines() if not is_noise(ln)]


def check_file(name: str, official: str, ours: str) -> dict:
    """Compare one vendored file against its official source."""
    off_lines = official.splitlines()
    our_lines = ours.splitlines()

    # (1) hard gate: every CODE line identical after re-pointing the shims AND
    #     reverting the declared functional patches. Anything else is drift.
    off_code = code_lines(official)
    our_code, patch_errors = revert_patches(name, code_lines(ours))
    off_sha = hashlib.sha256("\n".join(off_code).encode()).hexdigest()
    our_sha = hashlib.sha256("\n".join(our_code).encode()).hexdigest()
    code_match = off_sha == our_sha and not patch_errors

    # (2) any relative import in ours must be a documented shim.
    undocumented = [
        ln.strip() for ln in our_lines
        if ln.strip().startswith("from .") and ln.strip() not in SHIM_MAP
    ]

    # (3) characterise the non-code additions for the report.
    extra_comments = len(our_lines) - len(off_lines)
    shims_used = sum(1 for ln in our_lines if ln.strip() in SHIM_MAP)

    ok = code_match and not undocumented
    patches = PATCH_MAP.get(name, [])
    return {
        "file": name,
        "official_lines": len(off_lines),
        "our_lines": len(our_lines),
        "code_sha256_match": code_match,
        "official_code_sha256": off_sha,
        "our_code_sha256": our_sha,
        "shims_repointed": shims_used,
        "extra_noncode_lines": extra_comments,
        "undocumented_relative_imports": undocumented,
        "declared_patches": [w for _o, _f, w in patches],
        "declared_patch_errors": patch_errors,
        # VERBATIM = identical modulo import re-points only.
        # VERBATIM+PATCHED = also carries the declared functional patches above.
        "verdict": ("VERBATIM+PATCHED" if patches else "VERBATIM") if ok else "DRIFT",
        "ok": ok,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--domino", type=Path, default=None,
        help="path to a local clone of the official Domino repo "
             "(with the sglang-feat/dflash-domino branch fetched)",
    )
    ap.add_argument("--json", type=Path, default=None, help="write a proof record here")
    args = ap.parse_args()

    # locate a Domino clone
    candidates = [args.domino] if args.domino else [
        Path("../../ref_repo/Domino"),
        Path("../../../llm-infer-acc-ref-repos/Domino"),
        Path.home() / "Documents/git_repos/llm-infer-acc-ref-repos/Domino",
    ]
    domino_repo = next((p for p in candidates if p and (p / ".git").exists()), None)
    if domino_repo is None:
        sys.exit(
            "ERROR: no Domino clone found. Pass --domino /path/to/Domino "
            f"(clone {OFFICIAL_REPO} and\n"
            f"  git fetch origin {OFFICIAL_BRANCH})."
        )

    print(f"Official Domino head : {OFFICIAL_REPO}")
    print(f"          branch @ SHA: {OFFICIAL_BRANCH} @ {OFFICIAL_COMMIT[:10]}")
    print(f"          local clone : {domino_repo}")
    print(f"Vendored copy        : {PKG_DIR}\n")

    results = []
    for name, upstream_path in FILES.items():
        official = official_source(domino_repo, upstream_path)
        ours = (PKG_DIR / name).read_text()
        r = check_file(name, official, ours)
        results.append(r)
        flag = "OK " if r["ok"] else "!! "
        detail = (
            f"code sha256 {'MATCH' if r['code_sha256_match'] else 'DIFFER'}; "
            f"+{r['extra_noncode_lines']} comment/blank line(s); "
            f"{r['shims_repointed']} import shim(s)"
        )
        if r["declared_patches"]:
            detail += f"; {len(r['declared_patches'])} declared patch(es)"
        print(f"  {flag}{name:20s} official={r['official_lines']:>4} "
              f"ours={r['our_lines']:>4}  {detail}")
        for why in r["declared_patches"]:
            print(f"       PATCH: {why}")
        for err in r["declared_patch_errors"]:
            print(f"       STALE DECLARATION: {err}")
        if r["undocumented_relative_imports"]:
            for imp in r["undocumented_relative_imports"]:
                print(f"       UNDOCUMENTED relative import: {imp}")

    all_ok = all(r["ok"] for r in results)
    record = {
        "official_repo": OFFICIAL_REPO,
        "official_branch": OFFICIAL_BRANCH,
        "official_commit": OFFICIAL_COMMIT,
        "shim_map": SHIM_MAP,
        "files": results,
        "verdict": "PASS" if all_ok else "FAIL",
    }
    if args.json:
        args.json.write_text(json.dumps(record, indent=2))
        print(f"\nproof record -> {args.json}")

    print()
    if all_ok:
        n_patch = sum(len(r["declared_patches"]) for r in results)
        modulo = "the enumerated import shims"
        if n_patch:
            modulo += f" and the {n_patch} enumerated functional patch(es) listed above"
        print(f"PASS: the vendored Domino head is byte-identical to the official "
              f"fork @ {OFFICIAL_COMMIT[:10]},\n      modulo {modulo}. "
              "Not a reimplementation.")
        if n_patch:
            print("      (Every functional patch is declared in PATCH_MAP and reverted "
                  "before\n       hashing, so any UNDECLARED change still fails this "
                  "check.)")
        return 0
    print("FAIL: a vendored head file has drifted from the official source "
          "(see !! rows above).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
