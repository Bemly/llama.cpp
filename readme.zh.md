# llama.cpp RX 6800 / RDNA2 Metal 分支说明（`rx6800-fa-q8`，默认分支）

英文摘要见 [README.md](README.md) 顶部。本文件是完整中文版，与分支相对
`ggml-org/llama.cpp master` 的增量（14 个文件、18 个提交）逐项对应。

## 一、为什么自写 FA

- 上游 `FLASH_ATTN_EXT` 的 Metal 路径被 `has_simdgroup_mm` 门禁，RDNA2 没有
  matrix-core 快路径；实测 `simdgroup_matrix 8x8` 在本卡后端直接编译失败，
  上游 FA 在 RX 6800 上不可用，attention 回退到 CPU。
- 强行解禁上游 tile（`GGML_METAL_FA_ENABLE_AMD=1`）全面更差：短 ctx tg -58%，
  8k ctx tg -51% / pp -43%。标量回退吃掉全部 KV 红利。此开关留作负优化留档，
  默认关。
- 出路只有一条：自写 AMD 向量化 kernel，走 VALU `half2` packed FMA（实测持续
  20.4 TF，是 f32 的 2.7 倍）。本分支就是这条路，外加离散卡基建与 tile 调优。

## 二、新加的东西（按文件）

- `ggml/src/ggml-metal/kernels/fa_amd.metal`（新建，398 行）：decode vec
  （`nq==1`，tg128，KV 免 LDS 直读，Q half2 寄存器化，寄存器在线 softmax，
  split-k 自动 ~96 tg + reduce 合并）＋ prefill tile（`nq>1`，tg256，Q tile
  32 行，K 块过 LDS 转置 half4 布局并 +1 pad，逐行 8-lane 在线 softmax）。
  `dk==dv` ∈ {64,128,256}，NBC 64/128（dk256 只有 nbc64 朴素版）。
- `ggml-metal-ops.cpp`（+688）：`amd_supported` / `amd_quant_supported` 双门控、
  `DECODE/PP/FA` 三级相位旋钮、F16 与量化两套派发、scratch 预留与 4 个 encode
  位点的 dst 真 strides。
- `ggml-metal-device.m`（+117）：`supports_op` 里 `!has_simdgroup_mm` 时放行
  fa_amd 双门；混合 KV（K/V 类型不同）只给 AMD 分支开绿灯；RX 6800 编译宏；
  Private VRAM 镜像＋预算（`VRAM_BUDGET_MB` / `VRAM_RESERVE_MB` 默认 2048MB，
  `MMAP_PRIVATE_DISABLE` 可关）。
- `ggml-metal-device.cpp`（+379）：fa_amd 三套 pipeline getter、RX 6800 相位
  NR0/NSG 旋钮（RC2 移植）、`mul_mv` 变体选择。
- `ggml-metal-context.m`：独显默认关 concurrency（`CONCURRENCY_FORCE` 覆盖）、
  CB `NSError` 打印 domain/code/desc、`GGML_METAL_N_CB` 可配（MAX 8→16，默认仍
  1）、`DECODE_SCHED` 单 CB 折叠（默认关，实测 -4% tg，负优化留档）。
- `ggml-metal-impl.h`（+48）：kargs 加 `nbq_d/nbh_d/nbb_d`（dst 按 role 的真
  strides），reduce 复用 Q strides 处加注释＋双 encode 位点硬 assert。
- `kernels/dequantize.h`：`q4_0` / `q4_1` 4x4 按 CPU 逐字节重写（旧 `d/16`＋mask
  写法高半区少一次右移，直接错 16 倍）。
- `kernels/fa.metal`（+1 行）：补 `iq4_nl` 的 `kv_*_f16` 预实例化。
- `kernels/mul_mv.metal`（+86）：`q4_K/q6_K(_nr0_4/_nr0_8)`、
  `q8_0/q4_0/q5_0(_nr0_4)` 变体 kernel（`mul_mv_id` 的重复实例化被 metalc 拒掉，
  已砍；稠密模型不走 id 路径，无影响）。
- `tools/llama-bench/llama-bench.cpp`：`test_gen` 每 32 token 同步一次（原来每
  token 同步，14B tg 被低估 2.8%）。

## 三、门控规则（不满足就回退上游，一律安全默认关）

1. 总开关 `GGML_METAL_FA_AMD=1`（默认关）。
2. F16 路：K、V 同为 F16 且 `dk==dv` ∈ {64,128,256}。
3. 量化路：K/V 每侧独立取 {F16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL}（不许双 F16，
   双 F16 走第 2 条），形状同上；量化侧先 cast 到 scratch F16 再跑同一套 kernel。
4. 无 sinks、无 bias、无 softcap；Q/dst 行距与对齐护栏（float4/half4 读写要求）。

## 四、环境变量表

| 变量 | 默认 | 作用 |
|---|---|---|
| `GGML_METAL_FA_AMD` | 关 | 自研 FA 总开关 |
| `GGML_METAL_{DECODE_FA,PP_FA,FA}_{SPLIT,NBC}` | SPLIT 0=自动，NBC 128 | vec split 数（0=自动凑 ~96 tg，上限16）/ KV 块大小（dk256 钳 64） |
| `GGML_METAL_{DECODE,PP}_*_NR0/NSG`（+全局兜底） | nr0=2，nsg=4 | mul_mv 变体与 NSG 相位旋钮；Q8 `nr0=4` pp+28% 但 tg-3.6%，默认不动 |
| `GGML_METAL_N_CB` | 1（+主线程1个） | 大 graph 切分防 watchdog，上限 16；27B ub512 只有 16 能过 |
| `GGML_METAL_VRAM_BUDGET_MB` / `VRAM_RESERVE_MB` | 全 working set / 2048 | Private 镜像预算与保留 |
| `GGML_METAL_MMAP_PRIVATE_DISABLE` | 未设=开镜像 | 设了就关 Private 镜像 |
| `GGML_METAL_CONCURRENCY_FORCE` | 未设=独显关并发 | 强制开并发 |
| `GGML_METAL_FA_ENABLE_AMD` | 关 | 上游 FA 强上 AMD（负优化，留档） |
| `GGML_METAL_DECODE_SCHED` / `DECODE_STATS` | 关 | decode 单 CB 折叠（负优化，留档）/计数 |

decode-like 判据：第一个 MUL_MAT 的 `ne11==1`（FA 旋钮另要求 `ne12*ne13<=8` 的
轻 aux batch；tile 判据即 `nq==1`）。

## 五、实测（RX 6800，Metal，`-ngl 99`，单进程 `-r N`）

- Doujinshi-14B `Q4_K_M`：pp512 +4.4%、tg短 +2.9%、**pp8192 43.32→70.10
  （+61.8%）**、tg8k +5.4%；pp32768 上游 OOM，本分支 61.13 t/s（32k 解锁）。
- sakura-14B `Q6_K`：pp512 +2.7%、tg短 +5.0%、pp8192 +41.8%、tg8k +4.3%。
- Qwen3.5-0.8B `Q8_0`：基本持平（注意力占比太低，符合预期）。
- 27B（dk256，SSM 混血）：pp≤2k 持平（分母是 IQ3 GEMM＋SSM scan，都已在卡上，
  不是 attention 没上卡）；PPL F16 33.3418 vs 上游 33.3416（逐 chunk 对到 1e-3）。
- 正确性：`test-backend-ops -o FLASH_ATTN_EXT` mask=0 全绿（F16/全量化类型，
  vec＋tile）；mask=1 随机块 mask 残留 ERR≈0.026——half 精度的 adversarial
  fixture，全零/因果 mask 全过，定为 harness artifact，不是 kernel bug。
- **注水位声明**：上面的端到端数字测于 tile dst 槽位修复（`ba42cb1`）之前，
  修后必须重跑 PPL＋bench 重建，旧数在此之前引用即误导。

## 六、已知限制（都与 FA 无关，另起炉灶）

- 27B＋Metal 在 batch≥256 token 时 watchdog 超时（`decode prompt batch res=-3`，
  fa-off 同样挂；大 prompt GEMM 或 submission 切分问题），`N_CB=16` 绕过，
  或 `-b` 压到 ≤128。
- ATRI-7B F16 大 GEMM 同款 fault（F16＋Vulkan 直接 ErrorDeviceLost），跳过。
- sampler/argmax 与 embedding GET_ROWS 按设计留在 CPU（搬 embedding 上卡约
  0 收益、+400~600MB 显存，不值得）。

## 七、分支图

- `master`：上游基线（本地 718f7b4 一代）。
- `rx6800-tile`：RC2 二层 tile 调优（NR0/NSG）。
- `rx6800-fa`：FA-RDNA2 主线（dk128 时代）。
- `rx6800-fa-dk128`：dk128 验证分支。
- **`rx6800-fa-q8`：默认分支**，本说明描述的对象（含 Q8/Q4/Q5/iq4 全门＋DK256＋
  vec 双 fix＋dst-stride 修复 `ba42cb1`）。
- `rx6800-fa-q8-bak-20250915`：tile 根因定位前的快照备份。
- `rx6800-spec`：投机解码判定（ngram 无增益，draft 双墙判死，留档）。

## 八、用法

```sh
GGML_METAL_FA_AMD=1 ./build/bin/llama-server -m <14B模型> -ngl 99 \
  -c 65536 -fa on ...
```

F16 KV、dk=dv 128/64 自动接管，其余形状自动回退。单测：

```sh
GGML_METAL_FA_AMD=1 ./build/bin/test-backend-ops -o FLASH_ATTN_EXT -b MTL0
```
