# M3 Ultra prefill optimization plan

## Measured baseline (GLM-5.3-Flash Q4_K, 2x M3 Ultra 256GB, TB link)

| Config | Prefill tok/s | Decode tok/s | Notes |
|---|---|---|---|
| Solo single-node, q8_0 KV | 260-270 | 27 | model fits in ~171-180 GB |
| Solo single-node, f16 KV | 300 | 27 | FA-kernel dequant costs > KV bandwidth saves |
| 2-GPU RPC pipeline, 1 req | 286-291 | 15-23 | worker mostly idle (synchronous RPC) |
| 2-GPU RPC pipeline, 4 concurrent | 313 agg | - | no overlap; ~serialized |

Prefill is COMPUTE-bound (matmul FLOPs), not weight-bandwidth-bound.
The RPC pipeline is synchronous: head computes its layer block -> ships
activations -> worker computes. No overlap => 2nd GPU wasted.

## Config-layer wins (measured, now the default in the playbooks)
- f16 KV cache instead of q8_0: +11% prefill (300 vs 270 tok/s) with
  flash-attn on. q8_0's dequant path in the FA kernel costs more than the
  KV-memory bandwidth it saves at prefill.
- ubatch 2048: ~+4% prefill over 1024.
- Jumbo frames (MTU 9000) on the TB link: irrelevant now (cluster retired).

## MTP speculative decoding — ROOT CAUSE FOUND (not a code bug)

The draft-mtp crash ("backend is in error state from a previous command
buffer failure", solo AND over RPC) is GPU memory exhaustion:
kIOGPUCommandBufferCallbackErrorOutOfMemory. The MTP draft path's GPU
memory scales with slots x n_ctx:

| Config (GLM Q4_K, 512k ctx, M3 Ultra 256GB) | Result |
|---|---|
| 512k, 4 slots, f16 KV (even q8_0 KV both models) | GPU OOM at decode |
| 512k, 1 slot, default ~75% wired limit | GPU OOM (marginal) |
| 512k, 1 slot, wired limit 230GB (iogpu.wired_limit_mb=235520) | OK |
| 262k, 4 slots, default limit | OK |

Working config (now the default in the playbooks): -np 1 + f16 KV +
iogpu.wired_limit_mb=235520 -> decode 27 -> 35-37 tok/s (acceptance
~0.6-0.73, mean draft len ~3), prefill unaffected (295-307 tok/s).

Upstream-worthy finding: the per-slot draft-path memory at large n_ctx is
large enough that multi-slot serving OOMs a 256GB machine at 512k. A fix
would cap or share draft-path buffers across slots.

## Serving config now (262k, 4 slots)
Per user decision: context 262144 with FOUR slots (draft memory x 4 fits
at 262k; 512k only supports 1 slot). Verified: MTP decode 35.15 tok/s
(acceptance 0.61), prefill 312 tok/s.

## M3-Ultra hardware tuning (llama-bench sweep, GLM Q4_K, Apple9)
ubatch sweep at -fa 1, f16 KV:
  ubatch  512 -> pp4096 260.07 | tg256 27.19 tok/s
  ubatch 1024 -> pp4096 294.26 | tg256 27.24 tok/s
  ubatch 2048 -> pp4096 311.91 | tg256 27.23 tok/s
  ubatch 4096 -> pp4096 312.19 | tg256 27.26 tok/s
Prefill saturates at ubatch 2048; 4096 adds nothing. recommendedMax-
WorkingSetSize with iogpu.wired_limit_mb=235520: 246,960 MB.

## fa-vec kernel tuning on M3 Ultra (DONE — modest)
Sweep completed 2026-10-10 (f16, full head-size grid). 19 rows kept by the
no-harm compression; 15/19 IDENTICAL to the shipped pooled Apple9 rows —
the M3 Ultra mostly agrees with the family tuning; 4 rows differ. Table
patched on master (f70d3332e), numerics validated (240/240 on MTL0), and
the serving build now carries it.

Honest impact assessment for GLM (MLA dk=512 dv=512 f16): the one kept
row covers the deep-KV (>=16k) ne01==4 bucket at +7% on the fa_vec kernel
itself — attention is a slice of total decode cost, so end-to-end gain is
small and engages only at deep context with 4-row MTP steps. Short-KV
decode is baseline by design (bucket 0 always baseline). Measured runs:
35.3 tok/s short-KV (unchanged, expected), 40.1 tok/s on a deeper-context
run (partly acceptance variance 0.61 -> 0.74, not purely the tuning).

## mul_mm tile sweep on M3 Ultra (DONE — negative result, do not re-run)
Prefill is compute-bound at ~11 TFLOPS effective (~33% of the M3 Ultra's
fp16 peak) with Q4_K in-kernel dequant. Swept the compile-time mul_mm tile
constants (ggml-metal-impl.h N_MM_BLOCK_X/Y, N_MM_SIMD_GROUP_X/Y) on GLM:

  tile (BX,BY,SX,SY)   NRA x NRB / threads   pp4096 tok/s
  base 4,2,2,2 (ship)  128 x 256 / 128        312.19 +/- 0.18
  small 2,2,2,2        64 x 128 / 128        312.39 +/- 0.18
  wide 4,2,4,2        128 x 512 / 256        GPU fault (kIOGPUCommandBuffer-
                                              CallbackErrorInnocentVictim)
  tall 4,2,2,4        256 x 256 / 256        failed to run

Throughput is tile-INVARIANT across the legal envelope; larger tiles fault
the GPU (kernel geometry limits, not perf). The shipped (4,2,2,2) constants
are already optimal for M3 Ultra. The 33%-of-peak ceiling is the Q4_K
dequant ALU cost, not tiling — a mul_mm tuner would have nothing to win
here. Remaining prefill levers: smaller quant (Q2_K exists but 2-bit
quality cliff) or accept ~312 tok/s as practical for this model/quant.

Follow-ons worth doing:
- q8_0 sweep: gpt-oss-20b on Studio0 serves q8_0 KV -> fa_vec_q8_0.

## fa-vec kernel tuning on M3 Ultra (archived notes)
ggml-metal now ships a kernel tuner (tools/tuning, PR #27824). The
fa_vec_tuned_table keys rows by GPU FAMILY — Apple9 pools M3_ULTRA with
A18_PRO/M3_MAX/M3_PRO/M4/M4_MAX/M4_PRO, so no M3-Ultra-specific rows
exist. Sweep running on the M3 Ultra (f16 KV dtype, full head-size grid):
  ggml-metal-tuning fa-vec --dtype f16 > fa_vec_rows.txt 2> fa_vec_sweep.log
Next: retag the SKU column to family 9, replace the Apple9 f16 segment in
ggml-metal-tuning.cpp with the M3-Ultra rows (this fork targets M3 Ultra
only), validate numerics (test-backend-ops test -o FLASH_ATTN_EXT -b MTL0),
rebuild, and A/B decode. Note: fa-vec tunes the DECODE attention path only;
prefill matmul (mul_mm) has no tuner yet — a mul-mm tuner would be the
upstream-worthy fork contribution.

## Remaining fork work
1. Apply M3-Ultra fa-vec rows (sweep above) and A/B decode.
2. Pipeline micro-batching (Path B below) for 2-GPU prefill (deprioritized:
   the RPC cluster is retired).

## Paths
### Path A - Data parallelism (no fork, true 2x throughput)
Run one full solo instance per Studio, round-robin proxy in front.
~300 tok/s prefill each, ~600 aggregate. (Note: per user decision this
is NOT wanted — no load balancing; documented for completeness.)

### Path B - Pipeline micro-batching (fork, ~2x single-request latency)
Chunk a single prompt into K micro-batches and pipeline them so the head
computes chunk i while the worker computes chunk i-1, overlapping compute
and hiding the RPC transfer.
Design:
1. In ggml-rpc, add async compute: issue worker graph compute without
   blocking; return a handle; collect later.
2. In the llama decode path, split the prefill ubatch stream into
   micro-batches at the stage boundary; issue worker stage for mb[i] while
   local stage runs mb[i+1].
3. Double-buffer the boundary activation buffers.
Risk: high. Touches RPC + graph scheduling. Currently deprioritized —
the cluster is retired in favor of solo-on-Studio1.

## Current serving layout (playbooks @ db3a63b)
- Studio1: glm-5.3-flash solo :8081 (f16 KV, ubatch 2048, fa on, mmap)
- Studio0: gpt-oss-20b :8082, qwen3.8-flash-next (rapid-mlx) :8086
- ornith-1.5-35b track wired (8087) — weights not yet downloaded

## Recommendation
1. Fix the MTP draft Metal crash (biggest decode win, well-scoped bug).
2. Only revisit Path B if single-request prefill latency is still the
   hard requirement after that.
