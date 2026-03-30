#pragma once
#include "tensor.h"
#include <cmath>

//Computing cross-entropy loss for a single token prediction
inline float cross_entropy(const Tensor& logits, int target)
{
    int V = logits.cols;  // Vocabulary size
    // Step 1: Find maximum logit for numerical stability
    float max_val = -1e9;

    for (int i = 0; i < V; ++i)
    {
        // Use explicit if/else instead of std::max
        if (logits(0,i) > max_val)
        {
            max_val = logits(0,i);
        }
        else
        {
            // Keep max_val unchanged
            max_val = max_val;
        }
    }

    // Step 2: Compute denominator of softmax (sum of exp(logits - max))
    float sum = 0;
    for (int i = 0; i < V; ++i)
    {
        sum += std::exp(logits(0,i) - max_val);
    }
    // Step 3: Compute log probability of the target token
    float log_prob = logits(0,target) - max_val - std::log(sum);

    // Step 4: Cross-entropy loss = negative log probability
    return -log_prob;
}