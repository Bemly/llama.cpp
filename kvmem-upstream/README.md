# kvmem-upstream (vendored snapshot)

Self-contained copy of the KVMem sources this branch builds against, so the
fork does not depend on an external `kvmem-llama.cpp` checkout:

- `src/adapter/` — KVMem llama.cpp adapter (memory, batch, capture, Metal
  stage-in with the blit fast path; CUDA `.cu` kept for reference).
- `kvmem/` — host KVMem library (store, runtime, raw KV, rope) + tests.
- `tools/` — `llama-kvmem-cli` sources built by `tools/kvmem-cli`, plus
  `llama-kvmem-server.cpp`, `kvmem-vision.cpp` and `kvmem-*.h` built by
  `tools/kvmem-server`.

Snapshot source: local `kvmem-llama.cpp` at commit `81d03d8`
("Metal blit fast path (default on) + batched harvest D2H"), itself based
on upstream `kvmem/kvmem-llama.cpp`.

Build uses this copy by default (`LLAMA_KVMEM=ON` with no extra flags).
Override with `-DLLAMA_KVMEM_ROOT=/path/to/kvmem-llama.cpp` to build
against a live checkout instead.

To refresh the snapshot later: re-copy the three dirs from the checkout
and update the commit hash above.
