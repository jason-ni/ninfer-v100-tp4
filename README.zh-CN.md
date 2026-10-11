# NInfer V100 TP4

[English](README.md) · [详细评测方法](docs/performance.md) · [HTTP API](docs/serving.md)

面向 **4 × Tesla V100-SXM2 16 GB（`sm_70`，CUDA 12.8）** 的并发推理引擎，针对 Qwen3.8-27B。
TP4 是主要服务路径，TP2（两卡）作为回退，两条路径共用同一套引擎代码。默认配置使用官方 NVFP4
artifact（或 LM Studio GGUF 转换得到的 Q4_K_M `.ninfer`），180,000 token 容量，完整 INT8 group-64
KV，最多三个草稿的 MTP（允许接受零个）。

性能数据分为 [TP4](#tp4-测量首批数据) 和 [TP2 P2P 开启](#性能tp2--p2p-开启)/[P2P 关闭](#性能tp2--p2p-关闭) 两区；
构建与 [并发 HTTP 启动命令](#构建与模型) 在下方。

## 模型格式与来源

主要服务模型是
**Qwen3.8-27B-Coder390-EfficientThink-W4A4-W8A8-MTP-DFlash2**，来自
[nerkyor/Qwen3.8-27B-Coder390-EfficientThink-Opus5.5-GPT6Astra-Grok4.7-DSV4Pro-K3-SFT-RLOO-MTP-DFlash2](
https://huggingface.co/nerkyor/Qwen3.8-27B-Coder390-EfficientThink-Opus5.5-GPT6Astra-Grok4.7-DSV4Pro-K3-SFT-RLOO-MTP-DFlash2/tree/main/NVFP4-NInfer/W4A4-W8A8)
下的 `NVFP4-NInfer/W4A4-W8A8` NVFP4 容器
（`Qwen3.8-27B-Coder390-EfficientThink-W4A4-W8A8-MTP-DFlash2.ninfer`）。
容器内文本权重为 **4-bit FP4 (W4A4)** NVFP4（宽 prefill GEMM 路径），注意力 KV 为 **W8A8 INT8**，
原生 **MTP** 路径（≤3 token 草稿窗口），以及可选的五层 **DFlash2** 草稿包。

所有 vision-tower 权重——patch 投影、每层 block 的 qkv/o_proj/fc1/fc2，以及 merger 的 fc1/fc2——
均以**原生 FP16** 形式写入，由官方 BF16 vision safetensors 直接转换而来。其余所有 scale、bias
（包括 `vision/patch_embedding_bias` 保留为 BF16）、MTP 矩阵和前端资源均逐字节原样保留。
引擎实际加载的是重新导出的工作 artifact（`coder390-basemtp.ninfer`）。

FP16-vision 工作版 `.ninfer` **目前未单独发布**；HF 上的原始 NVFP4 容器是下载后直接使用的形式。
要复现已交付权重，需要由 AI 从原始 BF16 vision safetensors 转换：读取每个 vision linear
权重矩阵，BF16→FP16 cast，写入 artifact 为 `FP16` 对象，再重新导出容器。

## 并发推理

`max_concurrency=8` 下通过有界 FIFO 准入（无抢占），最多 8 个请求并发运行。新请求只有当八个 slot
中的一个空闲时才会被准入；运行中的请求永远不会被逐出。混合 prefill/decode forward 消除了
`decode → prefill → decode` 的串行交替：当一个 round 内同时存在 prefill owner 和若干 decode-ready
行时，owner 的 prefill slab 和 decode 行共享一次 flat forward pass，linear 层只运行一遍，prefill
和 decode 的 token 在同一时间窗口内完成，不再占用两个独立的调度单元。

混合前缀缓存（`--prefix-cache`，可选）支持在 device 驱逐后从 pinned host 层恢复会话状态，长轮次多
轮对话可在 device 容量压力下保留而不必重新计算前缀。

TP4 下 MTP3 投机解码完全集成：verify slab 在 mixed unit 中同时有 prefill 和 decode 行时被折入，
否则作为独立 verify 步骤运行。

## Metrics

serve 在 `GET /metrics` 暴露 Prometheus 指标：

```text
ninfer_kv_pages{pool="main",kind="capacity"}  135869434
ninfer_kv_pages{pool="main",kind="mapped"}    135869434
ninfer_kv_pages{pool="main",kind="cached"}     9308386
ninfer_kv_pages{pool="main",kind="free"}     126561058
ninfer_kv_page_bytes{pool="main"}               18872576
ninfer_prefix_cache_enabled                        1
ninfer_prefix_cache_host_bytes{kind="pinned"}    29527900160
ninfer_prefix_cache_host_bytes{kind="used"}          69206016
ninfer_requests{state="running"}                      4
ninfer_computed_prefill_tokens_total               1499798
ninfer_committed_decode_tokens_total                73984
```

`ninfer_kv_pages` 和 `ninfer_kv_page_bytes` 分别是主池和后端池（MTP/DFlash）的页组容量与字节数；
前缀缓存的 device snapshot 槽位、pinned host 容量/占用和 slabs 均有对应指标；
每个请求的调度状态（running/prefilling/decode-ready/waiting）和累计 token 计数也同步暴露。

## TP4 测量（首批数据）

### 单请求 decode — NVFP4, INT8-64 KV, TP4 eager, MTP3

NVFP4 v3 artifact、TP4（`--devices 0,1,2,3`）、`--max-context 262144`、INT8-64 KV、三草稿窗口
MTP、CUDA Graph。一条单请求流：85,000-token 提示、512-token 输出。TTFT 从请求发出到首个 token；
decode 吞吐是 TTFT 之后的已提交 token 每秒数。

| Prompt tokens | 已提交 Decode tok/s¹ |
|---:|---:|
| 85,000 | **~270** |

¹ 首批单请求测量值，TP4 下的单请求 decode 速率。TP4 的 P2P/peer-off A/B 对照测试尚在进行中。

## 本分支的改动

- Q4_K_M prefill：协作解码 GGML-K 块，写入调用方 FP16 workspace，再交给 Volta CUTLASS
  Tensor Core GEMM；GDN 控制投影保留 FP32 输出。源 GGUF 的 Q4_K/Q6_K 编码和 scale 不变。
- NVFP4/FP8：宽 prefill 每次调用只解码一次权重，支持 TP2 预打包矩阵；窄投影走 QPN
  Tensor Core 路线。T≥2048 的宽 TP2 NVFP4 MLP 使用 256-column GEMM tile。
  SwiGLU 的 gate/up 中间值保留 FP32，仅最终激活写回 BF16。
  修复了预打包 FP8 词表头在单 token 和 chunk 尾部错误走 row-major GEMV 的问题。
- 长上下文注意力：共享 QK 分块和 softmax 权重，协作读取 split 向量，保留原有 FP32 累加顺序。
  不裁剪历史，不用检索子集代替完整注意力，不额外降低 KV 精度。
- DFlash 局部注意力：V100 在分片数量足够时采用每行一个 warp，并以 FP32 保存 split
  numerator；极短窗口仍走 CTA 路径。支持 2048/4096 窗口，主模型完整历史 KV 不变。
- GDN prefill：Q/K 每行只做一次 FP32 归一化，供状态 tile 复用；FP32 递推顺序不变。
  3072-token TP2 的 GDN scratch 为 24 MiB。
- TP2 通信：图可捕获的 UVA D2D copy 加双向事件顺序；启动时检查双向精确传输。
  支持 direct P2P 和经过验证的 CUDA-managed staging，不另设显式 pinned-host 通信分支。
- Volta 残差路径：Q5 与 BF16 TP2 row-shard 使用可在 SM70 执行的 kernel，并通过独立数值检查。
- NVFP4 v3：读取官方 `NINFER\x00\x03` 单文件容器，映射到已注册的
  `qwen3.8-27b/nvfp4` 身份，不重打包权重字节；使用容器原始 chat template。
  容器可选的五层 DFlash2 组件已绑定为 text-only TP2 实验路径。
- DFlash 双卡词表传输：greedy target 验证只交换各分片的 argmax 值和 ID；
  草稿 selector 合并各卡的精确 top-16，不搬运完整词表 logits。采样验证仍保留完整 logits 路径。
  V100 的窄 BF16 草稿投影直接读取 BF16，以 FP32 SIMT 累加，避免小矩阵的 CUTLASS 开销。
- 前缀复用：TP2 与 optimized MTP3 可以恢复当前前沿或完整 turn/response checkpoint，
  只计算后续新增提示；支持重复恢复、精确命中和提前停止后的继续生成。

## 性能：TP2 — P2P 开启

2026-10-01 实测，双卡各 300 W，PCIe 3.0 ×16、PHB 拓扑。重启后使用 `iommu=pt`，
两 GPU 的 IOMMU domain 均为 `identity`，NVIDIA P2P read/write 检查通过，
NInfer 自动启用已有 direct P2P 路径。**这是 PCIe P2P，不是 NVLink。**
该配置在本机通过验证，不代表所有主板或转接方案都能启用 P2P。
两区使用同一套启动命令；NInfer 在启动时依据实际双向通信检查自动选择路径。
`iommu=pt` 本身不是 P2P 成功证明，`nvidia-smi topo -m` 的 PHB 标签也不是。

### NVFP4 v3：实际上下文占用

官方 NVFP4 v3、TP2、完整 INT8 group-64 KV、greedy（temperature=0）、MTP3、
optimized draft head、CUDA Graph。容量统一 180000（实际 KV 分配 180032），prefill chunk=3072。
每档两轮冷提示，无前缀复用、无额外请求 warmup；图预热在计时之外。
每轮生成 513 token，其中首个来自 prefill，后续 **512 个是计时 decode 输出**。
吞吐为均值 ± 样本标准差。

| 实际输入 token | Prefill tok/s | 已提交 Decode tok/s | MTP 接受率 |
|---:|---:|---:|---:|
| 3072 | 1783.96 ± 43.38 | 105.48 ± 0.01 | 72.97% |
| 8192 | 1754.43 ± 18.24 | 114.49 ± 0.01 | 81.98% |
| 16384 | 1699.06 ± 9.73 | 108.29 ± 0.003 | 79.47% |
| 32768 | 1596.24 ± 2.94 | 101.63 ± 0.03 | 79.69% |
| 65536 | 1397.83 ± 0.72 | 88.34 ± 0.04 | 79.12% |
| 85000 | **1306.48 ± 3.95** | **83.18 ± 0.13** | **79.12%** |

每档两轮输出 IDs 和 MTP 统计一致，无 EOS/EOG，均完成指定窗口。
不同长度的提示保留完整任务及 assistant 后缀，但代码正文、接受率不同，
不能由 8K 档更快推断上下文越长越快。这些是固定窗口吞吐，不是完整代码任务得分。

85K 输入、180K 容量满足此前提出的 1000 prefill / 70 decode tok/s 目标。
同输入关闭 P2P 的控制组为 79.18 decode tok/s，开启后为 83.18，提升 **5.05%**；
两路径各两轮的所有 513 个输出 IDs 和 speculative 字段一致，每轮 accepted/drafted=360/455，
152 rounds。不能拿下方旧语料的 78.424 tok/s 计算严格的 P2P 因果增益。

85K resident 请求平均 prefill 65.061 秒、decode 6.155 秒、总计 71.220 秒。
模型加载 19.662 秒，只在进程启动时进行一次，不计入请求吞吐。
全部上下文的阶段时间、wall decode、接受数量和测量限制见[性能文档](docs/performance.md#p2p-enabled)。

### 合格的 NVFP4 prefill 更新

85K 输入、180K 容量、MTP3、chunk=2560 的同条件对照，各实现测两轮冷请求：
加宽 TP2 MLP tile 后，prefill **1297.05 ± 2.21 → 1311.92 ± 2.06 tok/s（+1.15%）**。
全部 513 个输出 IDs 和接受统计相同；decode 为 81.63 → 81.71 tok/s，不据此宣称稳定提升。
测试时桌面显存占用使更大 chunk 无法通过启动余量检查，因此采用 2560；
上面的旧 3072-chunk 表不是这项改动的配对控制组。独立数值判据未放宽，
本轮没有达到交付条件的新有损注意力入口。

### NVFP4 v3：最大容量，固定 512-token 输入

固定 512 输入、256 timed decode token（总输出 257）；prefill chunk=1024，其他推理配置不变。
每档丢弃一轮 warmup，再测三轮。这里 decode 使用**首 token 到请求结束的 wall time**，
而上一张表使用 Engine decode 阶段时间。

| 最大容量 | Wall decode tok/s | Prefill tok/s | MTP 接受率 |
|---:|---:|---:|---:|
| 1024 | 106.12 ± 0.01 | 1443.78 | 70.04% |
| 2048 | 106.12 ± 0.04 | 1438.44 | 70.04% |
| 4096 | 105.96 ± 0.02 | 1440.81 | 70.04% |
| 8192 | 105.98 ± 0.01 | 1442.96 | 70.04% |
| 16384 | 106.06 ± 0.04 | 1443.37 | 70.04% |
| 32768 | 106.00 ± 0.03 | 1443.77 | 70.04% |
| 65536 | 106.01 ± 0.06 | 1443.42 | 70.04% |

七档容量均精确分配；21 轮输出和 MTP 统计一致，每轮接受 173/247 drafts，83 rounds。
这是只有 512-token 实际占用的容量测试，不能将约 106 tok/s 当成 64K 实际占用的速度。

### 通信与推理回归

10 KiB BF16 allreduce，500 次 host-sync 测量：

| 路径 | 平均延迟 µs | p50 µs | p99 µs |
|---|---:|---:|---:|
| PCIe P2P 开启 | 26.6085 | 25.901 | 43.910 |
| P2P 关闭，verified staging | 47.7961 | 47.181 | 62.424 |

通信平均延迟降低 44.33%，**不等于整段推理快 44.33%**。两路径的精确传输、
不整齐 shape、guard 和 64 连续轮检查通过。

Q4_K_M 与 NVFP4 v3 的真实 MTP、前缀缓存回归通过：Graph/eager 输出、logits、接受状态和
复用前沿一致；两个 probe 的 MTP/plain 32-token 序列一致；各检查 64 个 teacher-forcing
位置，无分歧、worst emitted-logit deficit=0。前缀覆盖重复恢复、追加、前缀变更、
精确前沿、接受轮内提前停止、改写回复后的恢复和恢复后采样。

3274-token 提示的三轮冷/缓存 TTFT 中位数：Q4_K_M 为 3.33492 秒 / 16.8689 毫秒，
NVFP4 为 3.01954 秒 / 14.6475 毫秒。Q4 的两处缓存/全新 prefill 首次分歧对应
teacher-forced logit deficit 都为 0，满足原有 near-tie 判据，不能说所有冷/缓存序列逐位一致。
NVFP4 本次对应探针未报告这些分歧；测试判据没有放宽。

NVFP4 检查采用 Q4 既有 gate 的临时诊断副本，仅改权重身份断言，链接同一 Engine/Ops。
本次未因纯通信配置变化重跑完整 CUDA 数学套件。上述证据不是通用“智商无损”评分。

### DFlash2 v3：独立容量的实验路线

官方 v3 容器的五层草稿器已接入 text-only TP2。草稿投影保留容器中的 W8 格式、
辅助张量为 BF16；五层全部使用局部注意力，因此移除了不会读取的草稿 Full-KV 池。
主模型的完整历史 KV 没有裁剪。验证配置为 **98304 容量、1024 prefill chunk**；
不承诺两张 16 GB 卡能容纳 180K DFlash，180K 默认仍用 MTP3。

以下为本次交付实现、P2P 开启、INT8 KV、greedy、CUDA Graph 的单次测量，
使用同一份实际 85000-token 的 LRU 代码提示和 draft=7。图预热不计时，不复用前缀。

| 计时 Decode token 数 | Prefill tok/s | 已提交 Decode tok/s | 接受 / 提出草稿 | 接受率 |
|---:|---:|---:|---:|---:|
| 512 | 1185.52 | **99.93** | 422 / 623 | 67.74% |
| 2048，强制窗口 | 1182.63 | **95.87** | 1674 / 2612 | 64.09% |

每次请求还有一个 prefill 输出 token，不计入 decode。512 窗口没有 EOS/EOG；
2048 窗口在第 1150 个总输出 token 出现 `<|im_end|>`，因 benchmark 禁用了模型默认停止而继续生成。
后者是持续执行压力数据，**不是有效长代码完成速度**；两个窗口均不是完整任务评分。
与上一版 argmax-only 路径相比，全部 513 / 2049 个输出 IDs 及接受统计相同。
分片 top-16 的两次单次比较只提高约 0.6–0.8%，不足以证明稳定性能增益。
分片候选与 FP64 lattice oracle、真实 graph/eager 和 64 个 teacher-forcing 位置检查通过；
这些证据不等于通用模型能力无损的评测。

另一份 85K 高代码语料、98304 容量、1024 chunk 的单次强制 2048-token 窗口：
DFlash3/5/7 分别为 79.30/76.10/87.91 tok/s，MTP3 为 86.14。
第 942 个总输出 token 已出现 `<|im_end|>`（DFlash5 为第 917 个），后面仍被强制生成，
所以这些也是压力吞吐，不是有效完成速度。DFlash3/7 与 MTP 的输出 IDs 一致，DFlash5 不一致。
DFlash7 的完整 resident 请求反而更慢：95.40 秒，对比 MTP3 的 92.92 秒。
新测的 1024-token 窗口速度更低；试验性 draft=9/11/15 无收益，已撤回，27B 仍最多七个草稿。

较早的同容量 v3 测量：3072 输入的 DFlash7 为 143.06 tok/s（两轮）；
另一份 85K 代码语料的 DFlash7 为 73.38 tok/s、接受率 47.18%，匹配的 MTP3 为
83.13 tok/s、接受率 78.56%。它们早于本次词表传输优化，不是当前代码重测值。
语料、输出长度和接受率会影响吞吐，不能把 99.93 与另一份语料的 MTP 或 P2P 关闭数据直接算增益。
方法和复现命令见[性能文档](docs/performance.md#dflash2-v3-separate-capacity)。

### DFlash7 / MTP3：非代码、正常停止评测

同一官方 v3 模型，固定 **98304 容量、1024 prefill chunk、TP2、INT8 KV、greedy、
CUDA Graph**，禁用前缀复用，开启模型默认停止；共完成 **80 次请求**。
全部在首个模型结束标记处停止，没有计入 EOS 后续写。
下面 **D/M = DFlash7 / MTP3**，单位为已提交 decode tok/s。
原生短提示的故事/翻译/JSONL/逻辑题分别为 129/395/118/417 个输入 token，各测两次取均值；
其余每个任务/长度/后端只测一次，不能当作稳定均值。

| 实际输入 token | 故事 D/M | 翻译 D/M | JSONL D/M | 逻辑题 D/M |
|---:|---:|---:|---:|---:|
| 原生短提示 | 49.20 / 74.83 | 134.68 / 123.00 | 210.50 / 137.09 | 158.88 / 126.10 |
| 1024 | 51.24 / 76.66 | 127.44 / 117.58 | 206.30 / 135.98 | 157.87 / 125.01 |
| 2048 | 51.09 / 74.41 | 128.45 / 116.14 | 200.78 / 133.63 | 173.32 / 126.34 |
| 4096 | 47.60 / 70.98 | 116.31 / 113.49 | 201.09 / 133.82 | 162.94 / 125.98 |
| 8192 | 49.11 / 70.17 | 125.74 / 116.00 | 198.69 / 132.06 | 166.37 / 125.39 |
| 16384 | 44.80 / 68.16 | 112.08 / 110.15 | 192.60 / 128.33 | 161.27 / 120.91 |
| 32768 | 41.44 / 63.23 | 110.56 / 108.44 | 175.01 / 119.59 | 142.03 / 111.83 |
| 65536 | 36.02 / 55.63 | 85.29 / 89.32 | 148.14 / 104.67 | 122.20 / 98.14 |
| 85000 | 32.26 / 51.67 | 79.61 / 84.86 | 137.65 / 98.77 | 111.05 / 91.60 |

JSONL 在全部长度上均为 DFlash 更快；85K 的接受率 98.30%，decode 比同输出 MTP 快 39.37%。
故事接受率仅约 12–14%，DFlash 反而明显更慢。DFlash prefill 在本轮所有档位都更慢，
因此 decode 优势不代表完整请求更快：85K 逻辑题总耗时为 78.47 / 76.87 秒 D/M。

JSONL 精确字段/算术/顺序检查、逻辑题 CHECK 均通过；翻译只检查了章节和术语表结构。
部分故事违反字数或必含字面项要求，原生 DFlash 故事遗漏 `ORCHID-37`。
36 组任务/长度配对中 30 组输出 IDs 完全一致，85K 四组均一致；
不能由此宣称所有任务逐 token 一致或通用“智商无损”。
[完整 prefill、接受率、输出长度、请求耗时和结果限制](docs/performance.md#stop-aware-non-code-workloads-dflash7-versus-mtp3)
已逐项记录。

## 性能：TP2 — P2P 关闭

这里区分当前同输入控制组和开启 P2P 前的历史结果。原本的 translated `DMA-FQ` IOMMU
domain 阻止 direct P2P，CUDA 采用经过验证的 UVA D2D staging。当前控制组只在诊断进程
使用 CUDA capability-query shim，未更改系统配置；它不是公开 CLI 的 P2P 开关。

### 当前 NVFP4 v3 同输入控制组

与上方 85K P2P 测量使用相同输入、模型、sampling、KV、MTP、图和输出窗口，各测两轮：

| Prefill tok/s | 已提交 Decode tok/s | Wall decode tok/s | MTP 接受率 |
|---:|---:|---:|---:|
| 1297.94 ± 2.66 | 79.18 ± 0.014 | 79.151 | 79.12% |

开启 P2P 的严格同输入增益为 decode +5.0459%、prefill +0.6578%。

### 早期 NVFP4 数据

以下使用重启前语料，不能替代上面的同输入控制组：

| 模型 / 窗口 | Prefill tok/s | 已提交 Decode tok/s | MTP 接受率 |
|---|---:|---:|---:|
| v2，85K 输入、180K 容量、512 timed 输出 | 1279.44 ± 2.94 | 78.36 ± 0.04 | 79.12% |
| 官方 v3，同一旧 85K 窗口 | 1277.61 ± 3.12 | 78.424 ± 0.0004 | 79.12% |
| 官方 v3，512 输入、2K 容量、512 timed 输出 | 1344.24 ± 33.15 | 98.831 ± 0.027 | — |

官方 v3 还完成了 21-token、100% 接受率的兼容性 smoke，短窗口达到 120.66 tok/s；
这不是稳定长窗口吞吐，也不是 v3 量化本身带来大幅加速的证据。
早期固定 512 输入、256 timed 输出的容量测试如下，每档一轮 warmup 加三轮测量：

| 最大容量 | Wall decode tok/s | Prefill tok/s | MTP 接受率 |
|---:|---:|---:|---:|
| 8192 | 97.65 ± 0.08 | 1369.9 | 70.04% |
| 16384 | 97.67 ± 0.09 | 1370.9 | 70.04% |
| 32768 | 97.76 ± 0.11 | 1370.8 | 70.04% |
| 65536 | 97.55 ± 0.15 | 1369.9 | 70.04% |

### 早期 Q4_K_M 与 LM Studio

Q4_K_M、TP2、INT8 KV、180K 容量、prefill chunk=4096：8192 输入的一次冷 prefill
为 1672.9 tok/s，85000 输入的两轮均值为 1251.44 ± 2.87 tok/s。
85K 输入、MTP3、optimized draft head、CUDA Graph 的两个 128-token decode 窗口测得
**50.68 ± 0.04 tok/s**，MTP 接受率 78.07%。这不是 P2P 开启后的 Q4 decode 测量。

旧容量对照固定 512 输入、256 timed 输出，每档一轮 warmup、三轮测量，greedy、Q8/INT8 KV、
MTP 最大三最小零。NInfer 使用 TP2，LM Studio CUDA 2.33.0 使用自动双卡分配：

| 最大容量 | NInfer Q4_K_M decode tok/s | LM Studio Q4_K_M decode tok/s |
|---:|---:|---:|
| 1024 | 60.06 ± 0.02 | 62.67 ± 0.24 |
| 2048 | 60.12 ± 0.02 | 62.78 ± 0.05 |
| 4096 | 60.11 ± 0.03 | 62.69 ± 0.10 |
| 8192 | 60.13 ± 0.06 | 62.78 ± 0.07 |
| 16384 | 60.14 ± 0.04 | 62.63 ± 0.06 |
| 32768 | 60.16 ± 0.05 | 62.58 ± 0.004 |
| 65536 | 60.06 ± 0.05 | 62.49 ± 0.16 |

同一旧 85K 提示、Q4_K_M、180K 容量的 LM Studio 单次诊断为 35.4977 tok/s，
后端将容量取整到 180224；它不是重复对照。用户先前报告的 45/57 tok/s 缺完整测量元数据，
不当作本轮实测基线，更不能用 Q4_K_M 的 LM Studio 结果证明 NVFP4 的量化质量相同。

### 历史数值限制

当前 DFlash2 v3 数据在上方 P2P 开启区；没有本次 DFlash 的关闭 P2P 配对测量。
旧 DFlash 数据来自不同容器和布局，不能作为当前路线的速度基线。

早期 72.28、75.53、78.90 等 NVFP4 成绩来自修复前的 SwiGLU 中间值路径。
独立 FP64 数学 oracle 检查暴露了 FP8/NVFP4 过早写回 BF16 的误差；修复为 FP32 中间值后，
原判据通过且未放宽容差。旧输出会改变，所以这些成绩不是当前合格数值基线。
更快但改变累加顺序的实验也不是交付路线。详见[数值与历史 profiler 说明](docs/performance.md#p2p-disabled)。

## 构建与模型

需要 64 位 Linux、NVIDIA 驱动、CUDA 12.8、CMake 3.28+、C++20、Ninja、pkg-config、
FFmpeg 开发库及 libcurl。依赖版本不足时可用仓库脚本构建私有依赖：

```bash
tools/v100/build_dependencies.sh
PKG_CONFIG_PATH="$PWD/build/_deps/install/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}" \
cmake -S . -B build-v100 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build build-v100 -j2
```

构建并发按本机情况调整，保留交互余量，整机 CPU 使用率不要超过约 85%。
模型、原始语料和 profiler 文件是本地前提，不随代码仓库提交；C++ 产品只读取 `.ninfer`。
NVFP4 可使用 [Neroued 官方容器](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer)，
本机放在 `/Models/ninfer-V100X2/qwen3_8_27b_nvfp4.ninfer`。

需要从本地 LM Studio GGUF 转换 Q4_K_M 时，用含 NumPy 的 Python 3.11 环境，例如本机 `.venv`：

```bash
.venv/bin/python3 -m tools.convert.qwen3_8_27b.convert_gguf \
  --model /Models/LM-Studio-models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf \
  --mmproj /Models/LM-Studio-models/lmstudio-community/Qwen3.8-27B-GGUF/mmproj-Qwen3.8-27B-BF16.gguf \
  --out /Models/ninfer-V100X2/qwen3_8_27b_q4_k_m.ninfer
```

GGUF 衍生身份只支持 Text/MTP，保留的 Vision 对象仅用于验证，不代表可以输入图像。

## CLI 和 API 启动

#### 单请求 CLI

Q4_K_M，TP2（两卡，`0,1`），180K 容量，MTP3：

```bash
CUDA_VISIBLE_DEVICES=0,1 GGML_CUDA_PDL=0 dist/tp2 \
  --model <artifact> --tp 2 --kv-dtype int8 --prefill-chunk 4096 \
  --max-ctx 180000 --mtp-draft-tokens 3 --lm-head-draft \
  --prompt "用 Python 实现一个有容量上限的任务队列，并解释测试方法。" -ng 512 --no-thinking
```

TP4（四卡）换 `dist/tp4`、`--tp 4`、`--devices 0,1,2,3`。进程退出时 KV 池（容量/mapped/cached/free）
以 JSON 打印到 stderr。

#### 并发 HTTP serve（TP4 主路径）

4 个活跃 slot、16 路 pending 排队、28 GiB pinned-host 层、262,144 capacity、混合前缀缓存、CUDA Graph、MTP3：

```bash
./run-tp4.sh \
  --max-concurrency 4 --pending-timeout-ms 300000 \
  --prefix-cache --host-cache-mib 28192 --max-context 262144 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --cuda-graph --port 8080 --model-id qwen3.8-27b
```

流式 Chat Completion 请求：

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"解释 prefill 与 decode 的区别。"}],
       "max_tokens":512,"temperature":0,"reasoning_effort":"high","stream":true}'
```

KV 指标实时抓取：

```bash
curl -s http://127.0.0.1:8080/metrics | grep 'ninfer_kv_pages\|ninfer_prefix_cache_host'
```

TP2 NVFP4 serve（旧兼容路径，两卡，3,072-token chunk）：

```bash
NINFER_V100X2_ARTIFACT=/Models/ninfer-V100X2/qwen3_8_27b_nvfp4.ninfer \
NINFER_V100X2_PREFILL_CHUNK=3072 \
NINFER_V100X2_SERVE_TP=2 NINFER_V100X2_SERVE_DEVICES=0,1 \
./tools/v100/v100-serve.sh --model-id qwen3.8-27b --port 8080
```

`--default-max-tokens 65536` 是未指定输出长度时的默认值，仍受剩余上下文容量限制。
服务默认复用兼容前缀；`--no-prefix-reuse` 用于强制冷提示对照。
缓存只复用当前前沿或完整 checkpoint，不是无限 RAM KV 存储。
启动时固定能力与显存预算，请求不能临时开启未加载的执行能力。

OpenAI Chat Completions 请求示例：

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"解释 prefill 与 decode 的区别。"}],"max_tokens":512,"temperature":0}'
```

还提供 `/v1/responses`、`/v1/messages`、`/health` 和 `/v1/models`。
流式输出、tool calls、鉴权及状态语义见[HTTP 文档](docs/serving.md)，具体参数以程序 `--help` 为准。

## 致谢与许可

基础来自 [Neroued/ninfer](https://github.com/Neroued/ninfer)、RTX 3060 TP2 路线和
[geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100)。Volta NVFP4/TP2 行为参考并
对照过 [plus1998/Ninfer-V100-Duo](https://github.com/plus1998/Ninfer-V100-Duo)；prefill 调研参考
[1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM)。DFlash2 实验参考
[Inco AI](https://inco.ai/blog/dflash2/) 及其[草稿模型](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)。
[KVMem](https://github.com/kvmem/kvmem-llama.cpp) 已调研但未集成：按查询检索部分历史会改变
完整上下文注意力语义，本配置没有悄悄替换成这种近似方案。

模型来自 [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)，NVFP4 的混合 FP8/NVFP4
权重来自 [Unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)，容器由
[Neroued](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) 打包。

本项目采用 [Apache-2.0](LICENSE)；归属声明见 [NOTICE](NOTICE)，第三方依赖保留各自许可。
