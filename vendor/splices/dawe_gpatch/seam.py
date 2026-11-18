"""Host Seam: single adapter bridging `Moissanite-Labs/MDE` (`mep`) into `dawe_gpatch.slice`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

from mep import SPEC_VERSION
from mep.ethos import Ethos, load_ethos
from mep.integrity import confined_path
from mep.server import TOOLS as MEP_TOOLS
from mep.server import handle_jsonrpc_message as handle_mep_jsonrpc

from . import slice as gpatch

EthosInput = Union[Ethos, Path, str]


def _tool(name: str, desc: str, props: Dict[str, Any], req: Sequence[str]) -> Dict[str, Any]:
    return {
        "name": name,
        "description": desc,
        "inputSchema": {"type": "object", "properties": props, "required": list(req)},
    }


_STR = {"type": "string"}
_BOOL = {"type": "boolean"}
_LINK = {"type": "string", "enum": ["copy", "hardlink", "symlink"]}

WMCP_TOOLS = [
    _tool(
        "wmcp_checkpoint_identify",
        "Compute checkpoint layout digest, tensor fingerprint, and side-file hashes.",
        {"checkpoint_dir": _STR},
        ("checkpoint_dir",),
    ),
    _tool(
        "wmcp_patch_create",
        "Create a checkpoint-bound .gpatch sparse weight delta stamped with MEP Ethos and derivation lineage.",
        {
            "source_dir": _STR,
            "out_patch": _STR,
            "name": _STR,
            "description": _STR,
            "edited_dir": _STR,
            "edits": {"type": "object"},
            "derivation": {"type": "object"},
            "max_changed_fraction": {"type": "number"},
        },
        ("source_dir", "out_patch", "name"),
    ),
    _tool(
        "wmcp_patch_inspect",
        "Inspect a confined .gpatch manifest, Ethos lineage, and touched tensor rows.",
        {"patch_path": _STR},
        ("patch_path",),
    ),
    _tool(
        "wmcp_patch_verify",
        "Verify a confined checkpoint against a .gpatch source or result fingerprint.",
        {
            "patch_path": _STR,
            "checkpoint_dir": _STR,
            "expect": {"type": "string", "enum": ["source", "result"]},
            "full": _BOOL,
            "require_derivation": _BOOL,
        },
        ("patch_path", "checkpoint_dir"),
    ),
    _tool(
        "wmcp_patch_apply",
        "Apply a verified .gpatch from source_dir into a new confined out_dir.",
        {"patch_path": _STR, "source_dir": _STR, "out_dir": _STR, "full": _BOOL, "link": _LINK},
        ("patch_path", "source_dir", "out_dir"),
    ),
    _tool(
        "wmcp_patch_restore",
        "Restore a patched checkpoint byte-for-byte back to its source fingerprint.",
        {"patch_path": _STR, "patched_dir": _STR, "out_dir": _STR, "full": _BOOL, "link": _LINK},
        ("patch_path", "patched_dir", "out_dir"),
    ),
]
_WMCP_TOOL_NAMES = frozenset(t["name"] for t in WMCP_TOOLS)


def _coerce_ethos(ethos_or_root: Optional[EthosInput]) -> Ethos:
    if isinstance(ethos_or_root, Ethos):
        return ethos_or_root
    return load_ethos(Path.cwd() if ethos_or_root is None else Path(ethos_or_root))


def _require_confined(root: Path, relative: str) -> Path:
    target, err = confined_path(root, relative)
    if err is not None or target is None:
        raise ValueError(err or "escapes_root")
    return target


def build_ethos_lineage(
    ethos: Ethos,
    derivation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    lineage: Dict[str, Any] = {
        "mep_spec_version": SPEC_VERSION,
        "ethos_name": ethos.name,
        "ethos_documents": {doc.path: doc.sha256 for doc in ethos.documents},
    }
    if derivation is not None:
        lineage["derivation"] = derivation
    return lineage


def dispatch_wmcp_tool(
    name: str,
    args: Dict[str, Any],
    ethos_or_root: Optional[EthosInput] = None,
) -> Dict[str, Any]:
    ethos = _coerce_ethos(ethos_or_root)
    root = ethos.root

    if name == "wmcp_checkpoint_identify":
        ck_dir = _require_confined(root, str(args.get("checkpoint_dir", "")))
        ident = gpatch.identify(ck_dir)
        return {
            "checkpoint_dir": ck_dir.relative_to(root).as_posix(),
            "fingerprint": ident.fingerprint,
            "layout_sha256": ident.layout_sha256,
            "n_tensors": ident.n_tensors,
            "config": ident.config,
            "tokenizer": ident.tokenizer,
            "hashes": ident.hashes,
        }

    if name == "wmcp_patch_create":
        src_dir = _require_confined(root, str(args.get("source_dir", "")))
        out_patch = _require_confined(root, str(args.get("out_patch", "")))
        patch_name = str(args.get("name", "")).strip()
        if not patch_name:
            raise ValueError("missing_patch_name")
        deriv = args.get("derivation")
        if deriv is not None and not isinstance(deriv, dict):
            raise ValueError("invalid_derivation_metadata")

        kw = {
            "name": patch_name,
            "description": str(args.get("description", "")),
            "lineage": build_ethos_lineage(ethos, derivation=deriv),
            "max_changed_fraction": float(args.get("max_changed_fraction", 0.25)),
        }
        edited_rel, raw_edits = args.get("edited_dir"), args.get("edits")
        if (edited_rel is None) == (raw_edits is None):
            raise ValueError("provide_exactly_one_of_edited_dir_or_edits")

        if edited_rel is not None:
            patch = gpatch.create_from_dirs(
                src_dir, _require_confined(root, str(edited_rel)), out_patch, **kw
            )
        else:
            if not isinstance(raw_edits, dict) or not raw_edits:
                raise ValueError("invalid_edits_map")
            parsed = {
                str(t): ([int(i) for i in s["indices"]], bytes.fromhex(str(s["new_hex"])))
                for t, s in raw_edits.items()
            }
            patch = gpatch.create_from_edits(src_dir, parsed, out_patch, **kw)

        summary = gpatch.describe(patch)
        summary["patch_path"] = out_patch.relative_to(root).as_posix()
        return summary

    if name == "wmcp_patch_inspect":
        return gpatch.describe(
            gpatch.load_patch(_require_confined(root, str(args.get("patch_path", ""))))
        )

    if name == "wmcp_patch_verify":
        patch_path = _require_confined(root, str(args.get("patch_path", "")))
        ck_dir = _require_confined(root, str(args.get("checkpoint_dir", "")))
        expect = str(args.get("expect", "source"))
        full = bool(args.get("full", True))
        patch = gpatch.load_patch(patch_path)
        problems, _ = gpatch.check_checkpoint(patch, ck_dir, expect, full=full)
        if bool(args.get("require_derivation", False)):
            lineage = patch.manifest.get("source", {}).get("lineage", {})
            if not isinstance(lineage, dict) or not lineage.get("derivation"):
                problems.append("opaque delta: missing source.lineage.derivation recipe")
        return {
            "ok": not problems,
            "expect": expect,
            "checked": "full" if full else "fast",
            "problems": problems,
        }

    if name in ("wmcp_patch_apply", "wmcp_patch_restore"):
        patch_path = _require_confined(root, str(args.get("patch_path", "")))
        in_key = "source_dir" if name == "wmcp_patch_apply" else "patched_dir"
        in_dir = _require_confined(root, str(args.get(in_key, "")))
        out_dir = _require_confined(root, str(args.get("out_dir", "")))
        fn = gpatch.apply_patch if name == "wmcp_patch_apply" else gpatch.restore_patch
        res = fn(
            patch_path,
            in_dir,
            out_dir,
            full=bool(args.get("full", True)),
            link=str(args.get("link", "copy")),
        )
        res["out"] = out_dir.relative_to(root).as_posix()
        return res

    raise KeyError(f"unknown wmcp tool: {name}")


def _error_code_for(exc: BaseException) -> str:
    if isinstance(exc, gpatch.IncompatibleCheckpoint):
        return "incompatible_checkpoint"
    if isinstance(exc, gpatch.PatchError):
        return "patch_error"
    if isinstance(exc, (ValueError, KeyError)):
        return "invalid_or_unconfined_argument"
    return "internal_error"


def handle_wmcp_jsonrpc(
    msg: Dict[str, Any],
    ethos_or_root: Optional[EthosInput] = None,
) -> Optional[Dict[str, Any]]:
    msg_id = msg.get("id")
    if msg_id is None:
        return None

    method = msg.get("method", "")
    params = msg.get("params") or {}
    root = (
        ethos_or_root.root
        if isinstance(ethos_or_root, Ethos)
        else (Path.cwd() if ethos_or_root is None else Path(ethos_or_root))
    )

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"tools": list(MEP_TOOLS) + list(WMCP_TOOLS)},
        }

    if method == "tools/call" and params.get("name", "") in _WMCP_TOOL_NAMES:
        tool_name = params["name"]
        try:
            payload = dispatch_wmcp_tool(
                tool_name, params.get("arguments") or {}, ethos_or_root=ethos_or_root
            )
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(payload, indent=2)}],
                    "structuredContent": payload,
                    "isError": False,
                },
            }
        except Exception as exc:  # noqa: BLE001
            err = {"error": str(exc), "error_code": _error_code_for(exc), "tool": tool_name}
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(err)}],
                    "structuredContent": err,
                    "isError": True,
                },
            }

    return handle_mep_jsonrpc(msg, root=root)
  
