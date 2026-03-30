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
        if (logits(0,i) > max_val)
        {
            max_val = logits(0,i);
        }
        else
        {
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

// Cross-entropy WITH gradient for backpropagation
inline float cross_entropy_with_grad(
    const Tensor& logits,
    int target,
    Tensor& d_logits   // gradient output
)
{
    int V = logits.cols;  // Vocabulary size

    // Step 1: Finding max logit for numerical stability
    float max_val = -1e9;
    for (int i = 0; i < V; ++i)
    {
        if (logits(0,i) > max_val)
            max_val = logits(0,i);
    }

    // Step 2: Computing softmax probabilities
    std::vector<float> probs(V);
    float sum = 0;
    for (int i = 0; i < V; ++i)
    {
        probs[i] = std::exp(logits(0,i) - max_val);
        sum += probs[i];
    }

    for (int i = 0; i < V; ++i)
    {
        probs[i] /= sum;
    }

    // Step 3: Computing gradient of loss w.r.t logits
    // d_logits = softmax - one_hot(target)
    for (int i = 0; i < V; ++i)
    {
        d_logits(0,i) = probs[i];
    }

    d_logits(0,target) -= 1.0f;

    // Step 4: Computing loss
    float log_prob = logits(0,target) - max_val - std::log(sum);
    return -log_prob;
}