# sm70 优化对比与移植清单：ninfer-V100X2 vs ninfer-v100-tpx

本文记录 V100X2 与姊妹分叉 tpx 在 sm70 加速优化上的差异，作为 V100X2 的**可移植优化清单**（porting backlog）。

对比对象：

- **V100X2** = 本仓库 `/media/d500/build/repos/ninfer-V100X2`（基于 `geoffwatts/ninfer-v100` + RTX 3060 TP2 工作）
- **tpx** = `/media/zt/prj/ninfer-v100-tpx`（基于 `geoffwatts/ninfer-v100 @ b37d0dd`）

两个仓库是 `geoffwatts/ninfer-v100` 的独立分叉，Volta 内核数学核心（`nvfp4_volta_qpn_gemm.cuh`、`volta_mma.cuh`、m8n8k4 路线）完全一致，因此可以逐文件比较。

**结论：V100X2 有明显可移植的 sm70 优化空间。** 主要差距在单卡内核：V100X2 的 INT8 decode attention 仍停在自述的 "correctness-first / phase-1" 状态，而 tpx 已推进到 phase 2。

## 移植进展（V100X2, 2026-10-10）

已移植 **#1 decode attention v2、#5 split reduce、#6 的 mailbox allreduce + head row-gather**
（N-rank，覆盖 tp2/tp4；不是 tpx 的 2-rank/两进程版）。工作记录与验证细节见
`sm70-v2-mailbox-port-plan.md`。

V100X2 实测（NVFP4+FP16-vision 成品，TP2 devices 0,1，INT8-G64 KV，MTP3 + lm-head-draft，
eager，180000 capacity，chunk 3072，512-token 窗口，65,536-token prompt，2 reps）：

| 配置 | committed decode tok/s | 相对 baseline |
|---|---:|---:|
| 移植前 baseline | 112.74 ± 0.26 | — |
| 全部开启 | **131.33 ± 0.28** | **+16.5%** |
| `NINFER_SM70_ATTN_V2=0`（phase-1） | 115.55 ± 0.17 | +2.5% |
| `NINFER_TP_MAILBOX=0`（NCCL） | 129.35 ± 0.19 | +14.7% |

- #1 v2 attention：**+13.7%**（65K），8K 时 +1.6%，随上下文增长；prefill 不变。接受率与 round
  数在所有配置完全一致（0.9794871795，260/0）。
- mailbox：**+1.5%**（65K）。本机 NCCL 走 NVLink P2P 时小消息 kernel 只有 ~17 µs，而启动探测
  因串行化 host launch 报 ~23-37 µs，夸大了差距；nsys 显示真实节省 ~3-5 µs/消息
  （128-137 次/round）。tpx 的大幅收益来自其 NCCL 无 P2P 经主机内存的路径。
- #5 split reduce：约 +1%（由 baseline 与 phase-1/no-mailbox 之差扣除 mailbox 得到）。
- 本机 host mailbox 比 tpx 的 `/dev/shm` 版慢（~50-112 µs），因此设备端 NVLink mailbox 总是
  被探测选中；host 版仅作为无 P2P 时的回退。

#2 prefill flash 重调、#3 QPN 流水、#4 融合残差仍未移植（#3/#4 仅影响 nvfp4/fp8 路线）。
Q4_K_M 85K 验收数据未测：本机没有 `qwen3_8_27b_q4_k_m.ninfer` 与 85K 语料。

## 共同基线

- 相同的 `nvfp4_volta_qpn_gemm.cuh` 解码/scale 逻辑、`volta_mma.cuh`、m8n8k4 路线。
- 双方都有：QPN prepack、Wide prefill CUTLASS、INT8 G64 KV、TP2、CUDA Graph、MTP。

## tpx 有、V100X2 没有的 sm70 优化（按收益排序）

| # | 优化 | V100X2 现状 | tpx 做法与实测 |
|---|---|---|---|
| 1 | **INT8 长上下文 decode attention** | `src/ops/kernel/gqa_attention_decode_i8_tc_volta.cuh:90-95`：4 warp（`DimSplit=4`）、`Bc=16`，2 个 warp 算 QK 再共享 | `src/ops/softmax_attention/dense/causal_cache/small_t_i8_volta_v2.cuh:46-47`：**8 warp、`Bc=64`**，每 warp 只算自己 8 个 key 的 QK（零冗余），smem 交换 max/P，各 warp 做 D/8 列 PV；int8→fp16 用 PRMT；PV 直接 fp32 mma。186K/T=4 单次调用 **2.79→1.29 ms** |
| 2 | **D256 prefill flash 重调** | `src/ops/launcher/gqa_attention_volta_flash.cu:64` 仍 `ncols1=16`；vendored `third_party/llama_cpp_fattn/fattn-mma-f16.cuh:129` 仍是旧 128-thread/32-key 配置 | `ncols1=32` + 新配置 `(256,256,64, 256,1, 64,64,64,64, 1, false)`：131K/1024-token **119→88.6 ms，27.8→37.4 TFLOP/s**，端到端 186K prefill 455→379 s |
| 3 | **NVFP4 QPN decode 软件流水** | `nvfp4_volta_qpn_gemm.cuh` 主循环仍是单组、逐组 `__ldg` | 8 组一次发射 + 双缓冲（`NINFER_NVFP4_QPN_UNROLL`）：TP2 down 58.4→54.5 µs、gate/up 47.9→43.6 µs、单卡 5120×17408 96→88.4 µs（T=4） |
| 4 | **融合残差 epilogue + fp16 激活暂存** | `src/ops/linear_add/nvfp4/nvfp4_linear_add_plan.cpp` Volta 路线是 `LinearThenAdd`：先写 BF16 投影临时量，再单独 `residual_add`（~line 103） | `.cu` 新增 `launch_nvfp4_volta_qpn_fp16_residual`：激活先转一次 fp16，残差相加融进 QPN 输出阶段，省临时量和单独 pass，只 round 一次 BF16 |
| 5 | **small-T split reduce** | 归约线程仍按 `for(split=…; split+=blockDim)` 单发 | `src/ops/softmax_attention/dense/causal_cache/small_t.cuh`：空闲的 3/4 线程按 group 分摊 split、无分支 ×8 unroll：**29→13 µs (T=4)，21→9 µs (T=1)** |
| 6 | **TP2 小消息通信** | 只用 NCCL（`src/ops/common/allreduce.cu`），10 KiB allreduce 约 26.6 µs | 主机锁页 `/dev/shm` mailbox（40 KB 26→16 µs）；NVLink 机器再用 CUDA IPC 的**设备端 mailbox**（40 KB 7.8 µs / LL 版 5.6 µs）；LM/草稿 head 按词表切到两卡（拼接 335→27 µs，decode +9~10%） |

其他 tpx 优化：d256-h12-kv2（TP2 分片）attention 几何路由、12/2 的更长 KV split、`NINFER_MTP_ATTN_WINDOW` 旋钮（默认关，会掉接受率）。

## V100X2 有、tpx 没有的（不是一边倒）

Q4_K_M prefill（GGML-K 解码→FP16 workspace→CUTLASS）、DFlash2 五层 draft + 滑窗 attention、W4A4、Q5/W8 Volta 残差路线、TP4、v3 容器、原生 FP16 vision tower、prefix cache。这些是 tpx 侧的空缺。

## 建议

最值得移植、互不冲突的四项：**#1 decode attention v2、#2 prefill flash 重调、#3+#4 QPN 流水+融合残差、#5 split reduce**。#6 需要确认目标机是否愿意引入主机/NVLink mailbox（NCCL 路径仍在）。

注意事项：

- tpx 的收益数字来自 32 GB SXM2、nvfp4、TP2；V100X2 是 16 GB 卡且默认 Q4_K_M，具体幅度需在 V100X2 上实测。
- 方向一致（长上下文 decode 与 prefill 是瓶颈），所以优化空间确实存在。
