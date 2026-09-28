#!/usr/bin/env python3
"""P5 offline weight repack (NInfer-absorb): GGUF -> 16KB-aligned GGUF.

Same spec, same tensors, same order. Only two things change:
  1. `general.alignment` metadata -> new alignment (default 16384).
  2. Every tensor data offset re-laid to that alignment (zero padding).

All metadata/infobyte patches are same-size, tensor bytes stream-copied
verbatim: output is verifiably lossless (per-tensor sha256 with --verify).

Why: Metal `newBufferWithBytesNoCopy` (zero-copy mmap mapping) needs
page-aligned offsets. Stock GGUFs ship 32B alignment, so every tensor must
be copied into private buffers at load. The repacked file is the format
prerequisite for any future mmap-direct load path. On discrete GPUs with
`use_shared_buffers=false` the loader still copies (private buffers are
faster for compute), so expect load-time parity, not a speedup.
Gate: output loads, needle correct, cold-load A/B + size delta reported.

Usage:
  python3 convert_repack_metal.py in.gguf out.gguf [--alignment 16384] [--verify]
"""

import hashlib
import struct
import sys

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "gguf-py"))

MAGIC = b"GGUF"
ALIGN_KEY = "general.alignment"
CHUNK = 1 << 20

_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _str(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8")


def _wstr(out, s):
    b = s.encode()
    out.write(struct.pack("<Q", len(b)) + b)


def _copy_value(f, out, typ):
    if typ in _SIZES:
        b = f.read(_SIZES[typ])
        out.write(b)
        return b
    if typ == 8:
        n = struct.unpack("<Q", f.read(8))[0]
        out.write(struct.pack("<Q", n))
        b = f.read(n)
        out.write(b)
        return b
    if typ == 9:
        atype = struct.unpack("<I", f.read(4))[0]
        an = struct.unpack("<Q", f.read(8))[0]
        out.write(struct.pack("<I", atype) + struct.pack("<Q", an))
        for _ in range(an):
            _copy_value(f, out, atype)
        return None
    raise ValueError(f"bad metadata type {typ}")


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__)
        return 2
    src, dst = args[0], args[1]
    alignment = 16384
    verify = "--verify" in args
    for i, a in enumerate(args):
        if a == "--alignment":
            alignment = int(args[i + 1])
    assert alignment & (alignment - 1) == 0 and alignment >= 32

    from gguf import GGUFReader

    rd = GGUFReader(src)
    blobs = [(t.name, t.n_bytes, t.data_offset) for t in rd.tensors]
    print(f"tensors: {len(blobs)} total_bytes: {sum(b for _, b, _ in blobs)}")

    with open(src, "rb") as f:
        assert f.read(4) == MAGIC
        ver = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        assert ver == 3 and n_tensors == len(blobs)

        with open(dst, "wb") as out:
            out.write(MAGIC + struct.pack("<IQQ", ver, n_tensors, n_kv))
            patched = False
            for _ in range(n_kv):
                key = _str(f)
                _wstr(out, key)
                typ = struct.unpack("<I", f.read(4))[0]
                out.write(struct.pack("<I", typ))
                if key == ALIGN_KEY and typ == 4:
                    old = struct.unpack("<I", f.read(4))[0]
                    out.write(struct.pack("<I", alignment))
                    print(f"alignment: {old} -> {alignment}")
                    patched = True
                else:
                    _copy_value(f, out, typ)
            if not patched:
                _wstr(out, ALIGN_KEY)
                out.write(struct.pack("<I", 4) + struct.pack("<I", alignment))
                # Header n_kv must cover the appended key: patch it in place.
                cur = out.tell()
                out.seek(16)
                out.write(struct.pack("<Q", n_kv + 1))
                out.seek(cur)
                print(f"alignment: (added) -> {alignment}")

            off_fields = []
            for _ in range(n_tensors):
                name = _str(f)
                _wstr(out, name)
                ndim = struct.unpack("<I", f.read(4))[0]
                out.write(struct.pack("<I", ndim))
                for _ in range(ndim):
                    d = struct.unpack("<Q", f.read(8))[0]
                    out.write(struct.pack("<Q", d))
                typ = struct.unpack("<I", f.read(4))[0]
                out.write(struct.pack("<I", typ))
                f.read(8)  # old offset, replaced below
                off_fields.append(out.tell())
                out.write(struct.pack("<Q", 0))

            # New layout in tensor order.
            base = out.tell()
            pad = (-base) % alignment
            out.write(b"\x00" * pad)
            base += pad
            new_offs = []
            cursor = 0
            for (_, nbytes, _) in blobs:
                new_offs.append(cursor)
                cursor += nbytes + ((-nbytes) % alignment)

            for pos, new in zip(off_fields, new_offs):
                cur = out.tell()
                out.seek(pos)
                out.write(struct.pack("<Q", new))
                out.seek(cur)

            # Stream blobs with padding; sha both sides when verifying.
            sha_old, sha_new = hashlib.sha256(), hashlib.sha256()
            for (_, nbytes, old_abs), new in zip(blobs, new_offs):
                f.seek(old_abs)
                out.seek(base + new)
                left = nbytes
                while left > 0:
                    chunk = f.read(min(CHUNK, left))
                    if not chunk:
                        raise IOError("short tensor blob")
                    left -= len(chunk)
                    if verify:
                        sha_old.update(chunk)
                    out.write(chunk)
                    if verify:
                        sha_new.update(chunk)
                tail = (-nbytes) % alignment
                if tail:
                    out.write(b"\x00" * tail)
            if verify:
                print(f"sha_old={sha_old.hexdigest()[:16]} sha_new={sha_new.hexdigest()[:16]} "
                      f"match={sha_old.hexdigest() == sha_new.hexdigest()}")
            print(f"wrote {dst} base={base} total={base + cursor}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
