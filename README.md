# llama.cpp (RX 6800 / RDNA2 Metal fork)

> This fork targets AMD RX 6800 (RDNA2) on Metal. Default branch is `kvmem-eval`.
> Upstream README continues below. Full notes in Chinese: [readme.zh.md](readme.zh.md).

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## RX 6800 / RDNA2 Metal fork notes (`kvmem-eval`)

Self-written Metal flash attention for AMD RDNA2 plus discrete-GPU hardening.
Upstream gates `FLASH_ATTN_EXT` on `has_simdgroup_mm`, which RDNA2 lacks
(`simdgroup_matrix` 8x8 fails in the Metal compiler), so attention falls back
to CPU on this card. This branch replaces that path with VALU kernels.

### What is added (vs upstream master)

- `ggml/src/ggml-metal/kernels/fa_amd.metal` (new): decode vec kernel
  (`nq == 1`, split-k with reduce merge), prefill tile kernel (`nq > 1`),
  for `dk == dv` in `{64, 128, 256}`. Packed `half2` math, register-side
  online softmax, KV block over LDS with padded stride.
- Quantized-KV path: dequant prepass to scratch F16 (existing `kv_*_f16`
  kernels plus a new `iq4_nl` instantiation), then vec/tile on scratch.
  Each K/V side accepts `F16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL`
  (not both F16). Mixed pairs work.
- Gate (default on, `GGML_METAL_FA_AMD=0` opts out): no sinks/bias/softcap,
  stride and alignment guards. Anything outside the gate falls back to
  upstream. DK256 naive is token-identical to CPU full attention over 40
  greedy tokens (Bonsai-27B, q8 KV); with the gate off, quantized KV
  degrades (34 graph splits, PPL 4507 vs 3.9), so the gate stays on.
- Phase knobs: `GGML_METAL_{DECODE_FA,PP_FA,FA}_{SPLIT,NBC}`. `SPLIT=0`
  targets ~96 threadgroups automatically; `NBC` defaults to 128.
- Dequant fixes: `dequantize_q4_0` / `dequantize_q4_1` 4x4 rewritten
  byte-explicit to match the CPU rows (old `d/16` + mask form was wrong
  for the high half).
- RX 6800 tile tuning: phase-aware `NR0`/`NSG` env knobs with extra
  `mul_mv` variants (`Q8_0` `nr0=4` gains ~28% pp; default stays `nr0=2`).
- Discrete-GPU hardening: private VRAM mirror with budget
  (`GGML_METAL_VRAM_BUDGET_MB`, `GGML_METAL_VRAM_RESERVE_MB`, opt-out via
  `GGML_METAL_MMAP_PRIVATE_DISABLE`), concurrency off on dGPUs unless
  `GGML_METAL_CONCURRENCY_FORCE=1`, command-buffer `NSError`
  (domain/code/desc) logging, configurable `GGML_METAL_N_CB`
  (max 16, for GPU-watchdog slicing).
- Bench harness: `test_gen` syncs every 32 tokens instead of every token
  (per-token sync under-reports Metal decode throughput).
- Negative results kept behind off-by-default gates: upstream FA forced on
  AMD (`GGML_METAL_FA_ENABLE_AMD=1`, much slower) and single-CB decode
  scheduling (`GGML_METAL_DECODE_SCHED=1`, -4% tg).

### Measured (RX 6800, Metal, `-ngl 99`)

- 14B Qwen3 `Q4_K_M`: `pp8192` 43.3 -> 70.1 (+62%), short-ctx tg +3~5%;
  32k prefill unlocked at 61.1 t/s (non-FA path OOMs). Second 14B
  (`Q6_K`): `pp8192` +42%. Small/hybrid models: neutral (attention share
  too small to matter). Note: end-to-end re-verification after the tile
  store-index fix (`ba42cb1`) is still pending; treat these as pre-fix
  numbers.
- Correctness: `test-backend-ops -o FLASH_ATTN_EXT` mask=0 fully green on
  F16 and all quantized KV types; random-block-mask cases carry a known
  ~0.026 fixture artifact (half-dot adversarial input, also fails on the
  vec path; neutral/causal masks pass).

### Use

```sh
./build/bin/llama-server -m <14B-model> -ngl 99 \
  -c 65536 -fa on ...
```

Known limits: Metal faults on prompt batches >= 256 tokens for some large
models (GPU watchdog, unrelated to FA; raise `GGML_METAL_N_CB` toward 16);
sampler and embedding lookup stay on CPU by design.

### PQ2_0 ternary (PrismML port, Metal-only)

Runs ternary `PQ2_0` GGUFs (ggml type 142, group-128 2-bit, e.g.
Bonsai-2-27B). Stock llama.cpp rejects the type and has no Hadamard
activation runtime, so three pieces were ported from the PrismML fork:

- Core: type plumbing (`ggml.h`, traits, quant/dequant/vecdot incl. ARM
  NEON and x86 VNNI, ftype names), `prism.hadamard.*` metadata parsing,
  rotation/sign buffers, and graph wiring through `build_lora_mm`
  (plus a verifier that throws instead of silently computing wrong math).
- Metal: FWHT upgrade (f16 inputs, threadgroup-staged kernels up to
  width 8192, fused sign-flip support in the kernel), `PQ2_0` kernels
  for `mul_mv`/`mul_mm`/`ext`/`id`/`get_rows`/`cpy`, host dispatch and
  op-support entries, `N_R0=8/N_SG=2` defaults.
- Left out: `fwht_signed` fusion (this tree uses the newer fusion-table
  framework; unfused path is correct, fusion is perf-only) and `PTQ1_0`.

FA is required, not optional, for these models: the 16 full-attention
layers (head dim 256) spill to CPU without it.

```sh
./build-q8/bin/llama-server \
  -m Bonsai-2-27B-PQ2_0-CRACK.gguf -ngl 99 -c 32768 -fa on \
  --temp 1.0 --top-p 0.95 --top-k 20 --port 8082
```

Measured (RX 6800, Metal, `-ngl 99`, `llama-bench -p 512 -n 64 -r 2`,
F16 KV): FA off pp 75.07 / tg 17.37, FA on pp 80.31 (+7%) /
tg 31.29 (+80%). Correctness: identical greedy output with FA on/off;
`test-backend-ops -o MUL_MAT` green on Metal (1265) and BLAS (11).
Tuning knob: `GGML_METAL_{DECODE,PP}_PQ2_0_NSG`
(`GGML_METAL_PQ2_0_NSG` fallback).

### KVMem eval (`kvmem-eval`: this branch)

KVMem (KV-context virtualization, https://github.com/kvmem/kvmem-llama.cpp)
wired for Metal, evaluated with Bonsai-2-27B. Adapter sources are compiled
from `LLAMA_KVMEM_ROOT` (a local `kvmem-llama.cpp` checkout, CUDA upstream);
this branch holds the llama.cpp-side Metal pieces:

- `ggml-metal-device.{h,m}`: synchronous batched D2D blit
  (`ggml_metal_blit_batched`), private scratch alloc, shared staging alloc
  for harvest D2H. Same blit-encoder pattern as `buffer_cpy_tensor`.
- `llama-kvmem-stagein-metal.cpp` + adapter (in `LLAMA_KVMEM_ROOT`,
  not pushed here): layout gather/scatter via blit instead of the host
  round-trip, per-block batched harvest D2H with zero extra copies.
  Default on, `KVMEM_METAL_BLIT=0` restores the host fallbacks.

Build: `cmake -B build-kvmem-on -DLLAMA_KVMEM=ON
-DLLAMA_KVMEM_ROOT=../kvmem-llama.cpp`, target `llama-kvmem-cli`
(same flags as upstream: `--kvmem --kvmem-budget N --kv-dtype f16|q8_0`).

Measured (RX 6800, Metal, `-ngl 99`, Bonsai-27B, q8 KV, budget 512):

- Reselect layout 40ms -> 4ms (`layout_d2h+h2d`), retrieval total 89ms ->
  52ms; needle BLUEBIRD-7 correct at 2.5k and 7.3k prompts
  (`KVMEM_PERF=1` for the breakdown, `KVMEM_METAL_BLIT=0` for A/B).
- q8 KV + retrieval validated (P1-C): FA quant path engaged
  (`kv_q8_0_f16` prepass + `fa_amd` dk256 kernels, 65/65 layers on GPU).
- 7.3k needle vs full-context baseline: pp +14%, tg +40%, and the
  correct answer (baseline answers vaguely).

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
