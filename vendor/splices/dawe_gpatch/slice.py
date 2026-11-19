"""Donor Slice: `geodesic-patch/1` (`.gpatch`) minimal reachable AST from `bandr-ai/dawe`.

Donor: https://github.com/bandr-ai/dawe/blob/main/dawe/patch.py (Apache-2.0)
Severed: `dawe.telemetry`, `argparse`, `sys`, `mmap`, and unreachable `create_from_dense` CLI path.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import shutil
import struct
import tempfile
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

FORMAT = "geodesic-patch"
VERSION = 1
PROVENANCE_FILE = "geodesic_applied.json"
SUFFIX = ".gpatch"

DTYPE_SIZE: dict[str, int] = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U64": 8, "U32": 4, "U16": 2, "U8": 1, "BOOL": 1,
}
_UNSIGNED: dict[int, str] = {1: "B", 2: "H", 4: "I", 8: "Q"}
_BIG_ENDIAN: bool = array("H", b"\x01\x00")[0] != 1

TOKENIZER_FILES: tuple[str, ...] = (
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "special_tokens_map.json", "added_tokens.json", "chat_template.jinja", "chat_template.json",
)
CONFIG_FILES: tuple[str, ...] = ("config.json", "generation_config.json")
HASH_CHUNK = 16 << 20
DIFF_BLOCK = 1 << 20
Log = Callable[[str], None]


class PatchError(Exception):
    """Raised when a `.gpatch` file or safetensors shard fails structural validation."""


class IncompatibleCheckpoint(PatchError):
    """Raised when a checkpoint disagrees with a `.gpatch` fingerprint, layout, or side files."""


def _prod(shape: Iterable[int]) -> int:
    out = 1
    for d in shape:
        if d < 0:
            raise PatchError(f"negative dimension: {d}")
        out *= d
    return out


@dataclass(frozen=True)
class TensorInfo:
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int

    @property
    def numel(self) -> int:
        return _prod(self.shape)

    @property
    def itemsize(self) -> int:
        return DTYPE_SIZE[self.dtype]


@dataclass(frozen=True)
class Shard:
    path: Path
    data_start: int
    tensors: dict[str, TensorInfo]
    metadata: dict[str, str]


@dataclass
class Identity:
    layout_sha256: str
    fingerprint: str
    n_tensors: int
    config: dict[str, str]
    tokenizer: dict[str, str]
    hashes: dict[str, str]


@dataclass
class TensorEdit:
    name: str
    dtype: str
    shape: tuple[int, ...]
    indices: list[int]
    old: bytes
    new: bytes
    source_sha256: str = ""
    result_sha256: str = ""

    @property
    def itemsize(self) -> int:
        return DTYPE_SIZE[self.dtype]

    @property
    def numel(self) -> int:
        return _prod(self.shape)


@dataclass
class Patch:
    manifest: dict
    edits: list[TensorEdit]
    size_bytes: int = 0

    @property
    def name(self) -> str:
        return str(self.manifest["name"])

    @property
    def patch_id(self) -> str:
        return str(self.manifest["patch_id"])

    def changed(self) -> int:
        return sum(len(e.indices) for e in self.edits)


def read_header(path: Path) -> Shard:
    path = Path(path)
    if not path.is_file():
        raise PatchError(f"safetensors file not found: {path}")
    size = path.stat().st_size
    with path.open("rb") as f:
        head = f.read(8)
        if len(head) < 8:
            raise PatchError(f"{path}: truncated header")
        (n,) = struct.unpack("<Q", head)
        if n > size - 8 or n > (100 << 20):
            raise PatchError(f"{path}: invalid header size {n}")
        raw_hdr = f.read(n)
    try:
        header = json.loads(raw_hdr.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PatchError(f"{path}: malformed header JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise PatchError(f"{path}: header must be an object")
    meta = header.pop("__metadata__", {}) or {}
    if not isinstance(meta, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in meta.items()
    ):
        raise PatchError(f"{path}: invalid __metadata__")
    data_start = 8 + n
    tensors: dict[str, TensorInfo] = {}
    for name, entry in header.items():
        if not isinstance(entry, dict):
            raise PatchError(f"{path}: invalid tensor entry {name!r}")
        dtype, shape_raw, offsets = entry.get("dtype"), entry.get("shape"), entry.get("data_offsets")
        if dtype not in DTYPE_SIZE or not isinstance(shape_raw, list) or not isinstance(offsets, list) or len(offsets) != 2:
            raise PatchError(f"{path}: malformed descriptor for {name!r}")
        info = TensorInfo(dtype, tuple(int(d) for d in shape_raw), int(offsets[0]), int(offsets[1]))
        if info.start < 0 or info.end < info.start or (info.end - info.start) != info.numel * info.itemsize:
            raise PatchError(f"{path}: byte length mismatch for {name!r}")
        if data_start + info.end > size:
            raise PatchError(f"{path}: tensor {name!r} extends past EOF")
        tensors[name] = info
    return Shard(path, data_start, tensors, dict(meta))


def write_safetensors(
    path: Path,
    tensors: dict[str, tuple[str, Sequence[int], bytes]],
    metadata: dict[str, str] | None = None,
) -> None:
    path = Path(path)
    header: dict[str, object] = {"__metadata__": dict(metadata)} if metadata else {}
    offset = 0
    payloads: list[bytes] = []
    for name, (dtype, shape, raw) in tensors.items():
        if dtype not in DTYPE_SIZE or len(raw) != DTYPE_SIZE[dtype] * _prod(shape):
            raise PatchError(f"invalid byte buffer for {name!r} ({dtype} {list(shape)})")
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
        payloads.append(raw)
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    blob += b" " * (-len(blob) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for buf in payloads:
            f.write(buf)


class Checkpoint:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise PatchError(f"checkpoint directory not found: {self.root}")
        shards = sorted(self.root.glob("*.safetensors"))
        if not shards:
            raise PatchError(f"no *.safetensors files in {self.root}")
        self.shards = [read_header(p) for p in shards]
        self.where: dict[str, tuple[Shard, TensorInfo]] = {}
        for sh in self.shards:
            for name, info in sh.tensors.items():
                if name in self.where:
                    raise PatchError(f"duplicate tensor {name!r} in {self.root}")
                self.where[name] = (sh, info)

    def names(self) -> list[str]:
        return sorted(self.where)

    def info(self, name: str) -> TensorInfo:
        if name not in self.where:
            raise PatchError(f"tensor {name!r} not in {self.root}")
        return self.where[name][1]

    def span(self, name: str) -> tuple[Path, int, int]:
        if name not in self.where:
            raise PatchError(f"tensor {name!r} not in {self.root}")
        sh, info = self.where[name]
        return sh.path, sh.data_start + info.start, sh.data_start + info.end


def sha256_range(path: Path, start: int, end: int) -> str:
    h = hashlib.sha256()
    rem = end - start
    with path.open("rb") as f:
        f.seek(start)
        while rem > 0:
            chunk = f.read(min(HASH_CHUNK, rem))
            if not chunk:
                raise PatchError(f"{path}: unexpected EOF")
            h.update(chunk)
            rem -= len(chunk)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_range(path, 0, path.stat().st_size)


def _shape_str(shape: Sequence[int]) -> str:
    return ",".join(str(d) for d in shape)


def layout_digest(ck: Checkpoint) -> str:
    lines = [f"{n}\t{ck.info(n).dtype}\t{_shape_str(ck.info(n).shape)}" for n in ck.names()]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def fingerprint_of(ck: Checkpoint, hashes: dict[str, str]) -> str:
    lines = [f"{n}\t{ck.info(n).dtype}\t{_shape_str(ck.info(n).shape)}\t{hashes[n]}" for n in ck.names()]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def side_files(root: Path) -> dict[str, dict[str, str]]:
    cfg = {fn: sha256_file(root / fn) for fn in CONFIG_FILES if (root / fn).is_file()}
    tok = {fn: sha256_file(root / fn) for fn in TOKENIZER_FILES if (root / fn).is_file()}
    return {"config": cfg, "tokenizer": tok}


def identify(root: Path | str, log: Log | None = None) -> Identity:
    ck = Checkpoint(root)
    hashes: dict[str, str] = {}
    for name in ck.names():
        if log is not None:
            log(f"hashing {name}")
        hashes[name] = sha256_range(*ck.span(name))
    sides = side_files(ck.root)
    return Identity(layout_digest(ck), fingerprint_of(ck, hashes), len(ck.where), sides["config"], sides["tokenizer"], hashes)


def _idx_dtype(numel: int) -> str:
    return "U32" if numel <= (1 << 32) else "U64"


def _pack_idx(indices: Sequence[int], dtype: str) -> bytes:
    arr = array("I" if dtype == "U32" else "Q", indices)
    if _BIG_ENDIAN:
        arr.byteswap()
    return arr.tobytes()


def _unpack_idx(raw: bytes, dtype: str) -> list[int]:
    step = 4 if dtype == "U32" else 8
    if len(raw) % step != 0:
        raise PatchError("misaligned index buffer")
    arr = array("I" if dtype == "U32" else "Q")
    arr.frombytes(raw)
    if _BIG_ENDIAN:
        arr.byteswap()
    return arr.tolist()


def _canonical(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _payload_digest(edits: list[TensorEdit]) -> str:
    h = hashlib.sha256()
    for e in edits:
        h.update(_pack_idx(e.indices, _idx_dtype(e.numel)))
        h.update(e.old)
        h.update(e.new)
    return h.hexdigest()


def _validate_edit(e: TensorEdit) -> None:
    k = len(e.indices)
    if e.dtype not in DTYPE_SIZE or k == 0 or len(e.old) != k * e.itemsize or len(e.new) != k * e.itemsize:
        raise PatchError(f"malformed edit buffer for {e.name!r}")
    numel = e.numel
    prev = -1
    for idx in e.indices:
        if idx < 0 or idx >= numel or idx <= prev:
            raise PatchError(f"invalid or unsorted index {idx} in {e.name!r}")
        prev = idx


def write_patch(path: Path | str, manifest: dict, edits: list[TensorEdit]) -> Patch:
    out_path = Path(path)
    m = dict(manifest)
    m.pop("patch_id", None)
    m["patch_id"] = hashlib.sha256(_canonical(m)).hexdigest()
    tensors: dict[str, tuple[str, Sequence[int], bytes]] = {}
    for e in edits:
        _validate_edit(e)
        idx_dt = _idx_dtype(e.numel)
        k = len(e.indices)
        tensors[f"index/{e.name}"] = (idx_dt, [k], _pack_idx(e.indices, idx_dt))
        tensors[f"old/{e.name}"] = (e.dtype, [k], e.old)
        tensors[f"new/{e.name}"] = (e.dtype, [k], e.new)
    meta = {"format": f"{FORMAT}/{VERSION}", "geodesic_patch": json.dumps(m, sort_keys=True)}
    write_safetensors(out_path, tensors, metadata=meta)
    return Patch(m, edits, out_path.stat().st_size)


def load_patch(path: Path | str) -> Patch:
    patch_path = Path(path)
    sh = read_header(patch_path)
    raw_m = sh.metadata.get("geodesic_patch")
    if not raw_m:
        raise PatchError(f"{patch_path}: missing geodesic_patch metadata")
    manifest = json.loads(raw_m)
    if manifest.get("format") != FORMAT or manifest.get("version") != VERSION:
        raise PatchError(f"{patch_path}: unsupported format")
    check_m = dict(manifest)
    expect_id = check_m.pop("patch_id", None)
    if not expect_id or hashlib.sha256(_canonical(check_m)).hexdigest() != expect_id:
        raise PatchError(f"{patch_path}: patch_id digest mismatch")

    edits: list[TensorEdit] = []
    with patch_path.open("rb") as f:
        def get_buf(key: str) -> bytes:
            if key not in sh.tensors:
                raise PatchError(f"{patch_path}: missing {key!r}")
            inf = sh.tensors[key]
            f.seek(sh.data_start + inf.start)
            return f.read(inf.end - inf.start)

        for t in manifest.get("tensors", []):
            n = t["name"]
            idx_inf = sh.tensors.get(f"index/{n}")
            if idx_inf is None or idx_inf.dtype not in ("U32", "U64"):
                raise PatchError(f"{patch_path}: invalid index tensor for {n!r}")
            indices = _unpack_idx(get_buf(f"index/{n}"), idx_inf.dtype)
            if len(indices) != t["changed"]:
                raise PatchError(f"{patch_path}: changed count mismatch for {n!r}")
            e = TensorEdit(n, t["dtype"], tuple(int(d) for d in t["shape"]), indices, get_buf(f"old/{n}"), get_buf(f"new/{n}"), t["source_sha256"], t["result_sha256"])
            _validate_edit(e)
            edits.append(e)
    if _payload_digest(edits) != manifest.get("payload_sha256"):
        raise PatchError(f"{patch_path}: payload_sha256 mismatch")
    return Patch(manifest, edits, patch_path.stat().st_size)


def _runs(indices: Sequence[int]) -> Iterable[tuple[int, int, int]]:
    i, n = 0, len(indices)
    while i < n:
        j = i
        while j + 1 < n and indices[j + 1] == indices[j] + 1:
            j += 1
        yield indices[i], indices[j] + 1, i
        i = j + 1


def diff_regions(a: tuple[Path, int, int], b: tuple[Path, int, int], itemsize: int) -> list[int]:
    pa, sa, ea = a
    pb, sb, eb = b
    if ea - sa != eb - sb:
        raise PatchError("regions differ in size")
    total, block = ea - sa, max(1, DIFF_BLOCK // itemsize) * itemsize
    indices: list[int] = []
    code = _UNSIGNED[itemsize]
    with pa.open("rb") as fa, pb.open("rb") as fb:
        off = 0
        while off < total:
            n = min(block, total - off)
            fa.seek(sa + off)
            fb.seek(sb + off)
            xa, xb = fa.read(n), fb.read(n)
            if xa != xb:
                aa, ab = array(code), array(code)
                aa.frombytes(xa)
                ab.frombytes(xb)
                if _BIG_ENDIAN and itemsize > 1:
                    aa.byteswap()
                    ab.byteswap()
                base = off // itemsize
                indices.extend(base + i for i, (p, q) in enumerate(zip(aa, ab)) if p != q)
            off += n
    return indices


def gather(region: tuple[Path, int, int], itemsize: int, indices: Sequence[int]) -> bytes:
    path, start, _ = region
    parts: list[bytes] = []
    with path.open("rb") as f:
        for lo, hi, _ in _runs(indices):
            f.seek(start + lo * itemsize)
            parts.append(f.read((hi - lo) * itemsize))
    return b"".join(parts)


def sha256_with_edits(region: tuple[Path, int, int], itemsize: int, indices: Sequence[int], new: bytes) -> str:
    path, start, end = region
    total, chunk = end - start, max(1, HASH_CHUNK // itemsize) * itemsize
    h = hashlib.sha256()
    with path.open("rb") as f:
        off = 0
        while off < total:
            n = min(chunk, total - off)
            f.seek(start + off)
            buf = bytearray(f.read(n))
            lo = bisect.bisect_left(indices, off // itemsize)
            hi = bisect.bisect_left(indices, (off + n) // itemsize)
            for k in range(lo, hi):
                p = indices[k] * itemsize - off
                buf[p : p + itemsize] = new[k * itemsize : (k + 1) * itemsize]
            h.update(buf)
            off += n
    return h.hexdigest()


def touched_rows(e: TensorEdit) -> list[int] | None:
    if len(e.shape) < 2:
        return None
    w = _prod(e.shape[1:])
    return sorted({i // w for i in e.indices})


def _build(
    ck: Checkpoint,
    ident: Identity,
    diffs: dict[str, tuple[list[int], bytes]],
    *,
    name: str,
    description: str = "",
    lineage: dict | None = None,
    max_changed_fraction: float = 0.25,
    out: Path | str,
) -> Patch:
    edits: list[TensorEdit] = []
    result_hashes = dict(ident.hashes)
    for tname in sorted(diffs):
        raw_idx, raw_new = diffs[tname]
        info = ck.info(tname)
        sz = info.itemsize
        if len(raw_new) != len(raw_idx) * sz:
            raise PatchError(f"{tname}: new byte length mismatch")
        if not raw_idx:
            continue
        raw_old = gather(ck.span(tname), sz, raw_idx)
        kept = [
            (int(idx), raw_old[k * sz : (k + 1) * sz], raw_new[k * sz : (k + 1) * sz])
            for k, idx in enumerate(raw_idx)
            if raw_old[k * sz : (k + 1) * sz] != raw_new[k * sz : (k + 1) * sz]
        ]
        if not kept:
            continue
        if len(kept) > max_changed_fraction * info.numel:
            raise PatchError(f"{tname}: changed fraction exceeds {max_changed_fraction}")
        k_idx = [x[0] for x in kept]
        k_old = b"".join(x[1] for x in kept)
        k_new = b"".join(x[2] for x in kept)
        res_sha = sha256_with_edits(ck.span(tname), sz, k_idx, k_new)
        result_hashes[tname] = res_sha
        e = TensorEdit(tname, info.dtype, info.shape, k_idx, k_old, k_new, ident.hashes[tname], res_sha)
        _validate_edit(e)
        edits.append(e)
    if not edits:
        raise PatchError("no weights differ; nothing to patch")
    src_block: dict[str, object] = {
        "fingerprint": ident.fingerprint,
        "layout_sha256": ident.layout_sha256,
        "n_tensors": ident.n_tensors,
        "config": ident.config,
        "tokenizer": ident.tokenizer,
    }
    if lineage is not None:
        src_block["lineage"] = lineage
    manifest = {
        "format": FORMAT,
        "version": VERSION,
        "name": name,
        "description": description,
        "source": src_block,
        "result": {"fingerprint": fingerprint_of(ck, result_hashes)},
        "tensors": [
            {
                "name": e.name, "dtype": e.dtype, "shape": list(e.shape), "numel": e.numel,
                "changed": len(e.indices), "source_sha256": e.source_sha256, "result_sha256": e.result_sha256,
            }
            for e in edits
        ],
        "changed_weights": sum(len(e.indices) for e in edits),
        "payload_sha256": _payload_digest(edits),
    }
    return write_patch(out, manifest, edits)


def create_from_dirs(source: Path | str, edited: Path | str, out: Path | str, *, name: str, log: Log | None = None, **kw) -> Patch:
    ck_s, ck_e = Checkpoint(source), Checkpoint(edited)
    if layout_digest(ck_s) != layout_digest(ck_e):
        raise IncompatibleCheckpoint("source and edited layouts differ")
    ident = identify(source, log=log)
    diffs: dict[str, tuple[list[int], bytes]] = {}
    for tname in ck_s.names():
        idx = diff_regions(ck_s.span(tname), ck_e.span(tname), ck_s.info(tname).itemsize)
        if idx:
            diffs[tname] = (idx, gather(ck_e.span(tname), ck_e.info(tname).itemsize, idx))
    return _build(ck_s, ident, diffs, name=name, out=out, **kw)


def create_from_edits(source: Path | str, edits: dict[str, tuple[Sequence[int], bytes]], out: Path | str, *, name: str, ident: Identity | None = None, **kw) -> Patch:
    ck_s = Checkpoint(source)
    src_ident = ident or identify(source)
    return _build(ck_s, src_ident, {k: (list(i), b) for k, (i, b) in edits.items()}, name=name, out=out, **kw)


def check_checkpoint(patch: Patch, root: Path | str, side: str, *, full: bool = True, log: Log | None = None) -> tuple[list[str], Identity | None]:
    if side not in ("source", "result"):
        raise ValueError(f"invalid side: {side!r}")
    want_fp = patch.manifest[side]["fingerprint"]
    problems: list[str] = []
    ck = Checkpoint(root)
    src_meta = patch.manifest["source"]
    if layout_digest(ck) != src_meta["layout_sha256"]:
        problems.append("tensor layout digest mismatch")
    actual_sides = side_files(ck.root)
    for kind in ("config", "tokenizer"):
        if src_meta.get(kind, {}) != actual_sides[kind]:
            problems.append(f"{kind} side-file digest mismatch")
    for e in patch.edits:
        if e.name not in ck.where:
            problems.append(f"missing edited tensor {e.name!r}")
            continue
        info = ck.info(e.name)
        if (info.dtype, info.shape) != (e.dtype, e.shape):
            problems.append(f"tensor {e.name!r} dtype/shape mismatch")
            continue
        want_sha = e.source_sha256 if side == "source" else e.result_sha256
        if sha256_range(*ck.span(e.name)) != want_sha:
            problems.append(f"tensor {e.name!r} {side} sha256 mismatch")
    ident: Identity | None = None
    if full and not problems:
        ident = identify(ck.root, log=log)
        if ident.fingerprint != want_fp:
            problems.append(f"full checkpoint {side} fingerprint mismatch")
    return problems, ident


def verify_checkpoint(patch_path: Path | str, root: Path | str, expect: str = "source", *, full: bool = True, log: Log | None = None) -> list[str]:
    problems, _ = check_checkpoint(load_patch(patch_path), root, expect, full=full, log=log)
    return problems


def _place(src: Path, dst: Path, link: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if link == "copy":
        shutil.copy2(src, dst)
        dst.chmod(dst.stat().st_mode | 0o200)
    elif link == "symlink":
        os.symlink(src.resolve(), dst)
    elif link == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        raise ValueError(f"invalid link mode: {link!r}")


def _transform(patch: Patch, src_root: Path, out_root: Path, direction: str, *, full: bool, link: str, log: Log | None) -> dict:
    in_side, out_side = ("source", "result") if direction == "apply" else ("result", "source")
    problems, _ = check_checkpoint(patch, src_root, in_side, full=full, log=log)
    if problems:
        raise IncompatibleCheckpoint(f"checkpoint {src_root} incompatible ({in_side}): {'; '.join(problems)}")
    src_res = src_root.resolve()
    out_res = out_root.resolve() if out_root.exists() else out_root.parent.resolve() / out_root.name
    if src_res == out_res:
        raise PatchError("refusing in-place checkpoint modification")
    if out_root.exists():
        if not out_root.is_dir() or any(out_root.iterdir()):
            raise PatchError(f"output directory non-empty: {out_root}")
        out_root.rmdir()
    out_root.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".geodesic_tmp_{out_root.name}_", dir=str(out_root.parent)))
    ck_src = Checkpoint(src_root)
    touched = {ck_src.where[e.name][0].path.name for e in patch.edits}
    try:
        for item in sorted(src_root.iterdir()):
            if item.name == PROVENANCE_FILE or item.is_dir():
                continue
            _place(item, tmp / item.name, "copy" if item.name in touched else link)
        ck_tmp = Checkpoint(tmp)
        for e in patch.edits:
            spath, start, end = ck_tmp.span(e.name)
            payload = e.new if direction == "apply" else e.old
            sz = e.itemsize
            with spath.open("r+b") as f:
                for lo, hi, v_idx in _runs(e.indices):
                    f.seek(start + lo * sz)
                    f.write(payload[v_idx * sz : (v_idx + hi - lo) * sz])
            want_sha = e.result_sha256 if direction == "apply" else e.source_sha256
            if sha256_range(spath, start, end) != want_sha:
                raise PatchError(f"post-{direction} sha256 mismatch on {e.name!r}")
        if direction == "apply":
            prov = {
                "tool": f"{FORMAT}/{VERSION}",
                "operation": direction,
                "patch_name": patch.name,
                "patch_id": patch.patch_id,
                "input_fingerprint": patch.manifest[in_side]["fingerprint"] if full else None,
                "output_fingerprint": patch.manifest[out_side]["fingerprint"] if full else None,
                "fingerprint_verified": bool(full),
            }
            (tmp / PROVENANCE_FILE).write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.rename(out_root)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return {
        "operation": direction,
        "out": str(out_root),
        "changed_weights": patch.changed(),
        "tensors": [e.name for e in patch.edits],
        "output_fingerprint": patch.manifest[out_side]["fingerprint"],
        "fingerprint_verified": bool(full),
        "checked": "full" if full else "fast",
    }


def apply_patch(patch_path: Path | str, source: Path | str, out: Path | str, *, full: bool = True, link: str = "copy", log: Log | None = None) -> dict:
    return _transform(load_patch(patch_path), Path(source), Path(out), "apply", full=full, link=link, log=log)


def restore_patch(patch_path: Path | str, patched: Path | str, out: Path | str, *, full: bool = True, link: str = "copy", log: Log | None = None) -> dict:
    return _transform(load_patch(patch_path), Path(patched), Path(out), "restore", full=full, link=link, log=log)


def describe(patch: Patch) -> dict:
    m = patch.manifest
    return {
        "name": m["name"],
        "description": m.get("description", ""),
        "patch_id": m["patch_id"],
        "file_bytes": patch.size_bytes,
        "dense_bytes_if_whole_tensors": sum(e.numel * e.itemsize for e in patch.edits),
        "changed_weights": patch.changed(),
        "source_fingerprint": m["source"]["fingerprint"],
        "result_fingerprint": m["result"]["fingerprint"],
        "source_layout_sha256": m["source"]["layout_sha256"],
        "n_source_tensors": m["source"]["n_tensors"],
        "config": m["source"]["config"],
        "tokenizer": m["source"]["tokenizer"],
        "lineage": m["source"].get("lineage", {}),
        "tensors": [
            {
                "name": e.name, "dtype": e.dtype, "shape": list(e.shape), "changed": len(e.indices),
                "of": e.numel, "rows": touched_rows(e), "source_sha256": e.source_sha256, "result_sha256": e.result_sha256,
            }
            for e in patch.edits
        ],
  }
  
