# M3 Ultra prefill optimization plan

## Measured baseline (GLM-5.3-Flash Q4_K, 2x M3 Ultra 256GB, TB link)

| Config | Prefill tok/s | Notes |
|---|---|---|
| Solo single-node (1 GPU) | 260-270 | model fits in 171 GB |
| 2-GPU RPC pipeline, 1 req | 286 | worker mostly idle |
| 2-GPU RPC pipeline, 4 concurrent | 313 | no overlap; ~serialized |

Prefill is COMPUTE-bound (matmul FLOPs), not weight-bandwidth-bound.
The RPC pipeline is synchronous: head computes its layer block -> ships
activations -> worker computes. No overlap => 2nd GPU wasted.

## Two viable paths

### Path A - Data parallelism (no fork, true 2x throughput)
Run one full solo instance per Studio (model copied to both SSDs), put a
round-robin proxy in front. Each node ~270 tok/s; 2 concurrent requests
(one per node) => ~540 tok/s aggregate. Best for multi-user / batch.
Cost: 180 GB model copy to Studio1 + a tiny proxy. No code changes.

### Path B - Pipeline micro-batching (fork, ~2x single-request latency)
Chunk a single prompt into K micro-batches and pipeline them so the head
computes chunk i while the worker computes chunk i-1, overlapping compute
and hiding the RPC transfer.
Design:
1. In ggml-rpc, add async compute: issue worker graph compute without
   blocking; return a handle; collect later.
2. In the llama decode path, when prefilling, split the ubatch stream into
   micro-batches at the stage boundary; issue worker stage for mb[i] while
   local stage runs mb[i+1].
3. Double-buffer the boundary activation buffers.
Risk: high. Touches RPC + graph scheduling. Needs careful correctness +
memory double-buffering for 512k context.

## Recommendation
Start with Path A (immediate 2x throughput, zero risk). Pursue Path B only
if single-request prefill latency (not throughput) is the hard requirement.
