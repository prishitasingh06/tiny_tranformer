#pragma once
#include "tensor.h"
#include "linear.h"
#include <cmath>

// Rectified Linear Unit
// If x > 0 → return x
// If x <= 0 → return 0
inline float relu(float x)
{
    if (x > 0)
    {
        return x;
    }
    else
    {
        return 0;
    }
}

/*
Feedforward Neural Network (FFN) Layer:

Neural network layers are usually structured as:
Input layer → Hidden layer(s) → Output layer
Input layer: receives data (x of size d)
Hidden layer: does the internal computation, adding capacity to model complex patterns (hidden neurons)
Output layer: produces the final result (y of size d)
In this FFN code:
Linear fc1; // d → hidden
Linear fc2; // hidden → d
fc1 maps input of size d → size hidden
fc2 maps hidden → output of size d

So hidden is just the width of the “middle” layer, i.e., how many neurons the network uses to “think” about the input.

This struct represents a 2-layer feedforward neural network (FFN) block.

Mathematical formulation:

    output = fc2(ReLU(fc1(x)))

flow of logic-

1. Input:
   - x is a tensor of shape (1 × d)
   - Think of it as a single vector (e.g., one token embedding)

2. First linear layer (fc1):
   - For each hidden neuron i:
       val = sum over j of (fc1.W(i,j) * x(0,j))
   - Computes a dot product between the i-th row of weights and input vector
   - Result: hidden vector h of shape (1 × hidden)

3. Activation (ReLU):
   - h(0,i) = relu(val)
   - Removes negative values
   - Introduces non-linearity so the network can learn complex patterns

4. Second linear layer (fc2):
   - For each output neuron i:
       val = sum over j of (fc2.W(i,j) * h(0,j))
   - Another dot product: hidden vector → output vector
   - Result: output tensor of shape (1 × d)

Intuition:
- fc1 expands input into a higher-dimensional space (hidden)
- ReLU filters and adds non-linearity
- fc2 compresses back to original dimension (d)
*/
struct FFN
{
    Linear fc1; // d → hidden
    Linear fc2; // hidden → d
    int d, hidden;

    FFN(int d_model, int hidden_dim)
        : fc1(hidden_dim, d_model),
          fc2(d_model, hidden_dim),
          d(d_model),
          hidden(hidden_dim) {}

    Tensor forward(const Tensor& x)
    {
        std::cout << "[DEBUG] FFN forward called\n";
        Tensor h(1, hidden);
        for (int i = 0; i < hidden; ++i)
        {
            float val = 0;
            for (int j = 0; j < d; ++j)
            {
                val += fc1.W(i,j) * x(0,j);
            }
            h(0,i) = relu(val);
        }

        Tensor out(1, d);
        for (int i = 0; i < d; ++i)
        {
            float val = 0;
            for (int j = 0; j < hidden; ++j)
            {
                val += fc2.W(i,j) * h(0,j);
            }
            out(0,i) = val;
        }
        return out;
    }
};



/* Notes for anyone reading this file-
- I have not added any bias related logic yet. Only using weights as of now. 
(val += W*x + bias): future implementation plan

- And no batching as of now
Only processes one vector at a time (1 × d)

*/
