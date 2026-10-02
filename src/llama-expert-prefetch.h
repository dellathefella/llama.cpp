#pragma once

#include <string>

struct llama_model_loader;

// Flash-Next (qwen4exp) speculative expert prefetch.
//
// MoE expert weights are served from the page cache (NVMe-backed); the graph's
// first touch would otherwise stall on page faults serially. This issues
// POSIX_FADV_WILLNEED over the expert tensor ranges in layer order so pages
// arrive ahead of compute. The PLE / n-gram table is NOT touched here: it is
// index retrieval (kB-scale random reads) and stays lazy-mapped.
//
// Env:
//   LLAMA_EXPERT_PREFETCH=0          disable
//   LLAMA_EXPERT_PREFETCH_PACE_MS=N  sleep N ms between layers (use when the
//                                    expert set exceeds page-cache capacity,
//                                    so readahead stays ahead of compute
//                                    instead of evicting itself); 0 = issue
//                                    everything immediately (default)
void llama_expert_prefetch_start(llama_model_loader & ml, const std::string & arch);
void llama_expert_prefetch_stop();
