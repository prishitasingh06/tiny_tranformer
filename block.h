#pragma once
#include "tensor.h"
#include "multi_head_attention.h"
#include "kv_cache.h"
#include "rmsnorm.h"
#include "ffn.h"

// Single Transformer block which contains-
// 1. Multi-head self-attention
// 2. Feed-forward network (FFN)
// 3. Residual connections + RMS normalization
struct TransformerBlock 
{
    int d;              // Model dimension (embedding size)
    int num_heads;      // Number of attention heads
    FFN ffn;            // Feed-forward network

    // Constructor initializing dimensions and FFN
    TransformerBlock(int d_model, int heads)
        : d(d_model), num_heads(heads), ffn(d_model, 4*d_model) {}

    //Forward pass through the transformer block
    Tensor forward(Tensor& x, KVCache& cache) 
    {
        // ---- SELF-ATTENTION ----
        // multi_head_attention_kv:
        // Performs multi-head attention using:
        // Query = x, Key = x, Value = x (self-attention)
        // cache stores past key/value for efficient autoregressive decoding

        std::cout << "[DEBUG] TransformerBlock forward called\n";
        Tensor attn = multi_head_attention_kv(x, x, x, cache, num_heads);

        // RESIDUAL CONNECTION (Attention)
        // Add input x back to attention output (skip connection)
        for (int j = 0; j < d; ++j)
        {
            attn(0,j) += x(0,j);
        }

        //NORMALIZATION
        // rmsnorm:
        // Root Mean Square Normalization
        // Stabilizes values and improves training/inference
        rmsnorm(attn);

        //  FEED-FORWARD NETWORK
        // Expands dimension (d -> 4d) then projects back
        Tensor ffn_out = ffn.forward(attn);

        // RESIDUAL CONNECTION (FFN
        // Add attention output back to FFN output
        for (int j = 0; j < d; ++j)
        {
            ffn_out(0,j) += attn(0,j);
        }

        //  FINAL NORMALIZATION 
        // Another RMSNorm after FFN
        rmsnorm(ffn_out);

        // Return final output of the transformer block
        return ffn_out;
    }
};