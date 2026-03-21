// attention.h
#pragma once
#include "tensor.h"
#include <cmath>
#include <algorithm>

/*
    Computes scaled dot-product attention.

    Inputs:
    - Q (Query matrix): shape (n, d)
    - K (Key matrix):   shape (n, d)
    - V (Value matrix): shape (n, d)

    Steps:

    1. Compute attention scores (Q * K^T):
       - For each pair of tokens (i, j), compute the dot product
         between Q[i] and K[j].
       - Scale the result by 1 / sqrt(d) to stabilize gradients.

    2. Apply softmax (row-wise):
       - For each row i:
         a. Subtract the maximum value for numerical stability.
         b. Exponentiate all elements.
         c. Normalize by dividing by the sum so the row sums to 1.
       - This converts raw scores into attention probabilities.

    3. Compute weighted sum (scores * V):
       - Each output row i is a weighted sum of all value vectors V[j],
         using attention weights scores[i][j].

    Output:
    - out: shape (n, d), the result of attending over V using Q and K.
*/
Tensor attention(Tensor& Q, Tensor& K, Tensor& V) 
{
    int n = Q.rows; // n = number of tokens (sequence length)
    int d = Q.cols; // d =embedding dimension
    Tensor scores(n, n); //store attention scores between every pair of tokens

    // =========================
    // Step 1: Compute Q * K^T
    // =========================
    for (int i = 0; i < n; i++)  // iterate over query rows
    {
        for (int j = 0; j < n; j++)  // iterate over key rows
        {
            float dot = 0.0f;  //dot product accumulator
            for (int k = 0; k < d; k++) // Compute dot product between Q[i] and K[j]
            {
                dot += Q(i,k) * K(j,k);  // multiply element-wise and sum
            }
            scores(i,j) = dot / std::sqrt(d); //stabilizing training-->Scale by sqrt(d) to prevent large values 
        }
    }

    // =========================
    // Step 2: Apply softmax
    // =========================
    for (int i = 0; i < n; i++)  // process each row independently
    {
        //for numerical stability)-->Find max value in row
        float max_val = -1e9;
        for (int j = 0; j < n; j++)
        {
            max_val = std::max(max_val, scores(i,j));
        }

        // Computing exponentials and sum
        float sum = 0.0f;
        for (int j = 0; j < n; j++) 
        {
            scores(i,j) = std::exp(scores(i,j) - max_val); // Subtract max_val to avoid overflow in exp()
            sum += scores(i,j); // sum of exponentials
        }

        // Normalizing each value so the row sums to 1
        for (int j = 0; j < n; j++)
        {
            scores(i,j) /= sum;
        }
    }

    // =========================
    // Step 3: Multiply by V
    // =========================

    // Output tensor (same shape as Q: n x d)
    Tensor out(n, d);

    for (int i = 0; i < n; i++)     // for each output row
    {
        for (int k = 0; k < d; k++) // for each feature dimension
        {
            float val = 0;          // accumulator for weighted sum

            // Weighted sum over all values V[j]
            for (int j = 0; j < n; j++) 
            {
                                             // scores(i,j) = how much token i attends to token j
                val += scores(i,j) * V(j,k); // V(j,k) = value vector component
            }
            out(i,k) = val;// Store result
        }
    }
    return out; // Final attended output
}