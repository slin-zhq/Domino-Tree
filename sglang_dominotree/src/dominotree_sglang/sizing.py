"""Pure sizing helpers for separating a tree budget from draft depth."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional


@contextmanager
def draft_sized_server_args(
    server_args, draft_block_size: Optional[int]
) -> Iterator[Optional[int]]:
    """Temporarily size draft-worker construction by the drafter's depth.

    ``speculative_num_draft_tokens`` carries the target verify node count for
    DominoTree (tree budget plus root), but upstream DFLASH also uses it while
    constructing the draft worker.  Yield the preserved target node count while
    temporarily replacing the shared field with ``draft_block_size``.  Always
    restore it, including when draft-worker construction raises.
    """
    if server_args is None:
        yield None
        return

    raw_verify_num_nodes = getattr(
        server_args, "speculative_num_draft_tokens", None
    )
    verify_num_nodes = (
        int(raw_verify_num_nodes) if raw_verify_num_nodes is not None else None
    )
    block_size = int(draft_block_size) if draft_block_size else None
    changed = (
        verify_num_nodes is not None
        and block_size is not None
        and verify_num_nodes != block_size
    )
    if changed:
        server_args.speculative_num_draft_tokens = block_size
    try:
        yield verify_num_nodes
    finally:
        if changed:
            server_args.speculative_num_draft_tokens = verify_num_nodes
