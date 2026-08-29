from __future__ import annotations

import unittest
from types import SimpleNamespace

from dominotree_sglang.sizing import draft_sized_server_args


class DraftSizedServerArgsTest(unittest.TestCase):
    def test_preserves_budget_32_while_draft_init_sees_block_size_16(self):
        server_args = SimpleNamespace(speculative_num_draft_tokens=33)

        with draft_sized_server_args(server_args, 16) as verify_num_nodes:
            self.assertEqual(verify_num_nodes, 33)
            self.assertEqual(server_args.speculative_num_draft_tokens, 16)

        self.assertEqual(server_args.speculative_num_draft_tokens, 33)

    def test_restores_verify_node_count_when_draft_init_raises(self):
        server_args = SimpleNamespace(speculative_num_draft_tokens=33)

        with self.assertRaisesRegex(RuntimeError, "draft init failed"):
            with draft_sized_server_args(server_args, 16):
                raise RuntimeError("draft init failed")

        self.assertEqual(server_args.speculative_num_draft_tokens, 33)

    def test_default_budget_does_not_mutate_shared_server_args(self):
        server_args = SimpleNamespace(speculative_num_draft_tokens=16)

        with draft_sized_server_args(server_args, 16) as verify_num_nodes:
            self.assertEqual(verify_num_nodes, 16)
            self.assertEqual(server_args.speculative_num_draft_tokens, 16)


if __name__ == "__main__":
    unittest.main()
