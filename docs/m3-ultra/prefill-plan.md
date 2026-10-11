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

## Remaining fork work
1. Pipeline micro-batching (Path B below) for 2-GPU prefill (deprioritized:
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
