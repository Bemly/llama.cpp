# llama.cpp RX 6800 / RDNA2 Metal 分支说明（`kvmem-eval`，默认分支）

英文摘要见 [README.md](README.md) 顶部。本文件是完整中文版，与分支相对
`ggml-org/llama.cpp master` 的增量逐项对应；PQ2_0 部分见第九节。

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

## 三、门控规则（不满足就回退上游；默认开关见环境变量表）

1. 总开关 `GGML_METAL_FA_AMD`（默认开，`=0` 关闭）。DK256 朴素版已对 CPU
   全注意力 token-identical（Bonsai-27B q8 KV，40 token 贪心）；关闭时量化
   KV 即错（34 splits＋PPL 4507 对 3.9），故默认接管。
2. F16 路：K、V 同为 F16 且 `dk==dv` ∈ {64,128,256}。
3. 量化路：K/V 每侧独立取 {F16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL}（不许双 F16，
   双 F16 走第 2 条），形状同上；量化侧先 cast 到 scratch F16 再跑同一套 kernel。
4. 无 sinks、无 bias、无 softcap；Q/dst 行距与对齐护栏（float4/half4 读写要求）。

## 四、环境变量表

| 变量 | 默认 | 作用 |
|---|---|---|
| `GGML_METAL_FA_AMD` | 开 | 自研 FA 总开关（`=0` 关） |
| `GGML_METAL_MMQ_AMD` | 开 | VALU blocked mul_mm（IQ3_S/Q4_K prefill，`=0` 关） |
| `GGML_METAL_MMQ_MV` | 关 | IQ3_S matvec 移植版（正确但暂无增益，`=1` 开） |
| `GGML_METAL_{DECODE,PP}_IQ3S_{NR0,NSG}` | NR0 4 / NSG 2 | IQ3_S mul_mv 扫参（`nr0_2/8` 变体已备） |
| `KVMEM_METAL_BLIT` | 开 | KVMem Metal blit 快路径（`=0` 回 host 回退） |
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
- **`rx6800-fa-q8`**：Q8/Q4/Q5/iq4 全门＋DK256＋vec 双 fix＋dst-stride 修复
 （`ba42cb1`），上一代默认分支。
- **`rx6800-fa-q8-pq2`：默认分支**，= fa-q8 ＋第九节的 PQ2_0 ternary 移植
  （1 个提交，30 文件）。
- `rx6800-fa-q8-bak-20250915`：tile 根因定位前的快照备份。
- `rx6800-spec`：投机解码判定（ngram 无增益，draft 双墙判死，留档）。

## 八、用法

```sh
./build/bin/llama-server -m <14B模型> -ngl 99 \
```

F16 KV、dk=dv 128/64 自动接管，其余形状自动回退。单测：

```sh
./build/bin/test-backend-ops -o FLASH_ATTN_EXT -b MTL0
```

## 九、PQ2_0 ternary 适配（Prism 移植，Metal only）

背景：Bonsai-2-27B 这类 ternary 模型权重存在旋转基里（block 1024
Hadamard 折叠），原版没有对应类型（ggml type 142）也没有 activation
变换，官方说法是直接不可用。本节就是把 Prism fork 的这套搬过来，
只搬 PQ2_0（PTQ1_0 没要）。

搬了什么（30 文件，+1162）：
- core 类型全套：`ggml.h` 加类型＋ftype、`ggml.c` traits、`quants`
  quant/dequant、CPU vecdot（NEON 和 x86-VNNI 照抄 Prism，保证逐位
  一致）、`gguf` 名、`llama-model-loader` ftype 映射。
- Hadamard：`prism.hadamard.*` 元数据解析（白名单只放行 LLAMA/QWEN3/
  QWEN35/QWEN3NEXT 系）、rotation/sign 常驻 buffer、graph 在
  `build_lora_mm(/_id)` 中央注入变换＋embedding 逆变换，另有 graph
  verifier——缺变换直接抛错，不静默算错。
- Metal：FWHT 升级（f16 输入、宽块 TG kernel 到 8192、kernel 内融
  sign 位）、`mul_mv/mul_mm/ext/id/get_rows/cpy` 的 PQ2_0 kernel＋分发，
  `N_R0=8/N_SG=2` 默认。
- 没搬的：`fwht_signed` 融合（本树 fusion 已表驱动，旧写法接不上；
  非融合正确，融合只是省一次 elementwise）、PTQ1_0。

实测（RX 6800，Metal，`-ngl 99`，Bonsai-2-27B-PQ2_0-CRACK 7.2G，
`llama-bench -p 512 -n 64 -r 2`，F16 KV）：
- FA 关：pp 75.07 / tg 17.37——16 个 full-attn 层（head 256）spill 到
  CPU，tg 腰斩。
- FA 开：pp 80.31（+7%）/ tg 31.29（+80%）。所以跑这类模型 FA 必开，
  且 KV 必须 F16（`iq4_nl` 会让 FA 脱钩）。
- 正确性：FA 开关同题同答（Paris）；`test-backend-ops -o MUL_MAT`
  Metal 1265＋BLAS 11 全绿；0.8B 日常模型 smoke 正常。

用法：

```sh
./build-q8/bin/llama-server \
  -m Bonsai-2-27B-PQ2_0-CRACK.gguf -ngl 99 -c 32768 -fa on \
  --temp 1.0 --top-p 0.95 --top-k 20 --port 8082
```

扫参旋钮：`GGML_METAL_{DECODE,PP}_PQ2_0_NSG`（`GGML_METAL_PQ2_0_NSG`
兜底），NR0 先定死 8。DK256 验证已做：40 token 贪心与 CPU 全注意力
逐字一致；同二进制 PPL 开 3.9 / 关 4507（q8 KV，关=34 splits 坏路径）。

## 十、KVMem eval（本分支 `kvmem-eval`）

KVMem（KV 上下文虚拟化，https://github.com/kvmem/kvmem-llama.cpp）的
Metal 接线，Bonsai-2-27B 上跑通 retrieval 全链路。adapter 源码从
`LLAMA_KVMEM_ROOT`（本地 `kvmem-llama.cpp`，CUDA 上游）直接编译；
本分支只放 llama.cpp 侧的 Metal 件：

- `ggml-metal-device.{h,m}`：同步批量 D2D blit
  （`ggml_metal_blit_batched`）、private scratch、harvest D2H 用的
  shared staging；与 `buffer_cpy_tensor` 同一 blit 范式。
- stagein-metal＋adapter（已 vendor 到本分支 `kvmem-upstream/`，
  快照自 `kvmem-llama.cpp` @ `81d03d8`）：layout gather/scatter 走
  blit（原 host round-trip），harvest 按 block 批量 D2H、零多余拷贝。
  默认全开，`KVMEM_METAL_BLIT=0` 回 host 回退。

构建（自包含，不需额外 checkout）：`cmake -B build-kvmem-on
-DLLAMA_KVMEM=ON`，目标 `llama-kvmem-cli`（用法同上游：`--kvmem
--kvmem-budget N --kv-dtype f16|q8_0`）。传
`-DLLAMA_KVMEM_ROOT=...` 可改连 live checkout。

实测（RX 6800，Metal，`-ngl 99`，Bonsai-27B，q8 KV，budget 512）：

- reselect layout 40ms→4ms（`layout_d2h+h2d`），retrieval 总计 89ms→
  52ms；2.5k 与 7.3k prompt 的 BLUEBIRD-7 needle 均命中
  （`KVMEM_PERF=1` 看分项，`KVMEM_METAL_BLIT=0` 做 A/B）。
- q8 KV＋retrieval 验证（P1-C）：FA 量化路径接管
  （`kv_q8_0_f16` 预通道＋`fa_amd` dk256 kernel，65/65 层在 GPU）。
- 7.3k needle 对无 kvmem 全量基线：pp +14%、tg +40%，且答案正确
  （基线含糊其辞）。

注意：GPU mean-K kernel 与 capture 批化经实测为噪音级（capture 抓
的是 ubatch 切片，decode 仅 1 行），没做；tg 缺口主因是 reselect 按
query 摊销，长回答自动稀释。

## 十一、KVMem server（`llama-kvmem-server`，Qwen3.8-27B）

`tools/kvmem-server` 从 vendor 源码编 OpenAI 兼容 server（文本＋tools，
vision 走 `--mmproj`）。RX 6800 上用自带 MTP 头的
`Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ3_M` 跑通：

```sh
GGML_METAL_N_CB=16 ./build-kvmem-on/bin/llama-kvmem-server \
  -m Qwen3.8-27B-IQ3_M.gguf --port 18204 -c 4096 -ngl 99 -b 128 \
  --kvmem-budget 1024 --kvmem-gen-reserve 512 --kv-dtype q8_0 \
  --spec-type draft-mtp --kvmem-mtp-state snapshots
```

Metal 必备（已在树内改默认）：`--kvmem-mtp-state snapshots`（replay
只要 CUDA，直接拒绝启动）、MTP KV 继承 `--kv-dtype`（F16 MTP KV 在
Metal 下算出垃圾）、`-b 128`＋`GGML_METAL_N_CB=16`（大 prefill 图触
watchdog）。端到端验证：2.5k prompt needle 经 HTTP 返回 BLUEBIRD-7，
MTP 接受率 ~67%。

## 十二、MMQ for Metal（`GGML_METAL_MMQ_AMD`，默认开）

上游 dense `mul_mm` 要 `simdgroup_mm`（Apple 独占），RDNA2 上 dense
prefill 全走逐行 `mul_mv`（权重每行重读一遍，Qwen3.8-27B IQ3_M 只有
pp 4.25）。`mul_mm.metal` 加 VALU blocked kernel
（`kernel_mul_mm_{iq3_s,q4_K}_f32_amd`，64x64 tile，K 步长 32，在线
dequant，不要 tensor core）：pp 4.25→42.7，同卡 stock
Vulkan/MoltenVK 是 25.1，已超。门控：IQ3_S/Q4_K＋F32 activation＋
K%256==0＋M>8 行；`=0` 可关。`test-backend-ops -o MUL_MAT` 3/3＋端到
端 needle 背书。

decode（`mul_mv`）另算：原版 Metal tg 3.65 对 Vulkan 9.15；移植版
（`kernel_mul_mv_iq3_s_f32_mmq`，`GGML_METAL_MMQ_MV=1` opt-in）正确
（套件全绿）但性能持平，先保持 opt-in。旋钮全扫过（`IQ3S_NR0/NSG`、
`nr0_2/8` 变体），tg 波动 <5%——差的是结构不是参数。0.8B 小模型
tg Metal 176 对 Vulkan 157，无固定开销问题。
