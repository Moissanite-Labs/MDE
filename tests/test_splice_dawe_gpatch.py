"""Lock contract test suite for `vendor.splices.dawe_gpatch` using native `MDE` inputs."""

from __future__ import annotations

import json
import shutil
import struct
import tempfile
import unittest
from pathlib import Path

from mep.ethos import Ethos, load_ethos
from vendor.splices.dawe_gpatch.seam import handle_wmcp_jsonrpc
from vendor.splices.dawe_gpatch.slice import sha256_file, write_safetensors


def _pack_bf16(values: list[int]) -> bytes:
    return struct.pack(f"<{len(values)}H", *values)


def _make_mde_workspace(root: Path) -> Ethos:
    (root / "principles").mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("# Model Domain Expansion\n", encoding="utf-8")
    (root / "principles" / "ethos.md").write_text(
        "# No Opaque Deltas\nEvery weight artifact binds to its checkpoint and derivation.\n",
        encoding="utf-8",
    )
    (root / "ethos.json").write_text(
        json.dumps({"name": "mde-q2-harness", "learning": "active"}),
        encoding="utf-8",
    )
    ckpt = root / "checkpoints" / "qwen_toy_v1"
    ckpt.mkdir(parents=True, exist_ok=True)
    write_safetensors(
        ckpt / "model-00001-of-00002.safetensors",
        {"model.embed_tokens.weight": ("BF16", [4, 4], _pack_bf16([0x3C00 + i for i in range(16)]))},
    )
    write_safetensors(
        ckpt / "model-00002-of-00002.safetensors",
        {"model.norm.weight": ("F32", [2, 2], struct.pack("<4f", 1.0, 0.0, 0.0, 1.0))},
    )
    (ckpt / "config.json").write_text('{"architectures":["Qwen3ForCausalLM"]}\n', encoding="utf-8")
    (ckpt / "tokenizer.json").write_text('{"version":"1.0"}\n', encoding="utf-8")
    (ckpt / "chat_template.jinja").write_text("{{ messages[0]['content'] }}\n", encoding="utf-8")
    return load_ethos(root)


class DaweGpatchLockTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.ethos = _make_mde_workspace(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _rpc(self, msg_id: int | None, method: str, params: dict | None = None) -> dict | None:
        msg: dict = {"jsonrpc": "2.0", "method": method}
        if msg_id is not None:
            msg["id"] = msg_id
        if params is not None:
            msg["params"] = params
        return handle_wmcp_jsonrpc(msg, ethos_or_root=self.ethos)

    def test_mep_v01_delegation_and_tool_listing(self) -> None:
        resp = self._rpc(1, "tools/list")
        assert resp is not None
        names = {t["name"] for t in resp["result"]["tools"]}
        self.assertIn("ethos_capabilities", names)
        self.assertIn("wmcp_patch_create", names)

        cap = self._rpc(2, "tools/call", {"name": "ethos_capabilities", "arguments": {}})
        assert cap is not None
        self.assertEqual(cap["result"]["structuredContent"]["mep_version"], "0.1")

    def test_roundtrip_apply_restore_and_derivation_gate(self) -> None:
        s1 = self.root / "checkpoints" / "qwen_toy_v1" / "model-00001-of-00002.safetensors"
        orig_bytes = s1.read_bytes()

        create = self._rpc(
            10,
            "tools/call",
            {
                "name": "wmcp_patch_create",
                "arguments": {
                    "source_dir": "checkpoints/qwen_toy_v1",
                    "out_patch": "patches/eos12.gpatch",
                    "name": "eos12",
                    "derivation": {"compiler": "ridge_readout_v1", "lambda_ridge": 10.0},
                    "edits": {
                        "model.embed_tokens.weight": {
                            "indices": [8, 9, 10, 11],
                            "new_hex": _pack_bf16([0xC000, 0xC100, 0xC200, 0xC300]).hex(),
                        }
                    },
                },
            },
        )
        assert create is not None
        self.assertFalse(create["result"]["isError"])
        meta = create["result"]["structuredContent"]
        self.assertEqual(meta["changed_weights"], 4)
        self.assertEqual(meta["lineage"]["ethos_name"], "mde-q2-harness")
        self.assertEqual(meta["lineage"]["derivation"]["compiler"], "ridge_readout_v1")

        ver = self._rpc(
            11,
            "tools/call",
            {
                "name": "wmcp_patch_verify",
                "arguments": {
                    "patch_path": "patches/eos12.gpatch",
                    "checkpoint_dir": "checkpoints/qwen_toy_v1",
                    "expect": "source",
                    "require_derivation": True,
                },
            },
        )
        assert ver is not None
        self.assertTrue(ver["result"]["structuredContent"]["ok"])

        app = self._rpc(
            12,
            "tools/call",
            {
                "name": "wmcp_patch_apply",
                "arguments": {
                    "patch_path": "patches/eos12.gpatch",
                    "source_dir": "checkpoints/qwen_toy_v1",
                    "out_dir": "checkpoints/patched",
                },
            },
        )
        assert app is not None and not app["result"]["isError"]
        patched_s1 = self.root / "checkpoints" / "patched" / "model-00001-of-00002.safetensors"
        self.assertNotEqual(sha256_file(patched_s1), sha256_file(s1))

        rest = self._rpc(
            13,
            "tools/call",
            {
                "name": "wmcp_patch_restore",
                "arguments": {
                    "patch_path": "patches/eos12.gpatch",
                    "patched_dir": "checkpoints/patched",
                    "out_dir": "checkpoints/restored",
                },
            },
        )
        assert rest is not None and not rest["result"]["isError"]
        restored_s1 = self.root / "checkpoints" / "restored" / "model-00001-of-00002.safetensors"
        self.assertEqual(restored_s1.read_bytes(), orig_bytes)

    def test_template_drift_and_opaque_delta_rejection(self) -> None:
        edt = self.root / "checkpoints" / "edited"
        shutil.copytree(self.root / "checkpoints" / "qwen_toy_v1", edt)
        vals = [0x3C00 + i for i in range(16)]
        vals[0] = 0xBEEF
        write_safetensors(
            edt / "model-00001-of-00002.safetensors",
            {"model.embed_tokens.weight": ("BF16", [4, 4], _pack_bf16(vals))},
        )
        self._rpc(
            20,
            "tools/call",
            {
                "name": "wmcp_patch_create",
                "arguments": {
                    "source_dir": "checkpoints/qwen_toy_v1",
                    "edited_dir": "checkpoints/edited",
                    "out_patch": "patches/opaque.gpatch",
                    "name": "opaque",
                },
            },
        )
        ver_opaque = self._rpc(
            21,
            "tools/call",
            {
                "name": "wmcp_patch_verify",
                "arguments": {
                    "patch_path": "patches/opaque.gpatch",
                    "checkpoint_dir": "checkpoints/qwen_toy_v1",
                    "require_derivation": True,
                },
            },
        )
        assert ver_opaque is not None
        self.assertFalse(ver_opaque["result"]["structuredContent"]["ok"])

        (self.root / "checkpoints" / "qwen_toy_v1" / "chat_template.jinja").write_text("DRIFT\n", encoding="utf-8")
        app_drift = self._rpc(
            22,
            "tools/call",
            {
                "name": "wmcp_patch_apply",
                "arguments": {
                    "patch_path": "patches/opaque.gpatch",
                    "source_dir": "checkpoints/qwen_toy_v1",
                    "out_dir": "checkpoints/fail",
                },
            },
        )
        assert app_drift is not None
        self.assertTrue(app_drift["result"]["isError"])
        self.assertEqual(app_drift["result"]["structuredContent"]["error_code"], "incompatible_checkpoint")

    def test_confinement_rejects_escape_paths(self) -> None:
        esc = self._rpc(
            30,
            "tools/call",
            {"name": "wmcp_checkpoint_identify", "arguments": {"checkpoint_dir": "../escape"}},
        )
        assert esc is not None
        self.assertTrue(esc["result"]["isError"])
        self.assertEqual(esc["result"]["structuredContent"]["error"], "absolute_empty_or_parent_path")


if __name__ == "__main__":
    unittest.main()
