#pragma once
#include "tensor.h"
#include "linear.h"

// Backprop through LM head
inline void lm_head_backward(
    const Tensor& x,
    Linear& lm_head,
    const Tensor& d_logits,
    Tensor& d_x
)
{
    int vocab = lm_head.W.rows;
    int d = lm_head.W.cols;

    //Gradient for weights
    for (int v = 0; v < vocab; ++v)
    {
        for (int j = 0; j < d; ++j)
        {
            lm_head.W.grad[v*d + j] += d_logits(0,v) * x(0,j);
        }
    }

    //Gradient for input
    for (int j = 0; j < d; ++j)
    {
        float val = 0;
        for (int v = 0; v < vocab; ++v)
        {
            val += d_logits(0,v) * lm_head.W(v,j);
        }
        d_x(0,j) += val;
    }
}

// SGD weight update
inline void update_weights(Tensor& W, float lr)
{
    for (int i = 0; i < W.data.size(); ++i) 
    {
        W.data[i] -= lr * W.grad[i];
        W.grad[i] = 0.0f;
    }
}
