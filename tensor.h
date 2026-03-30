// tensor.h
#pragma once
#include <vector>

struct Tensor {
    std::vector<float> data;
    int rows, cols;
    std::vector<float> grad; //for gradient storage

    Tensor(int r, int c) : rows(r), cols(c), data(r * c), grad(r*c, 0.0f) {}

    // Read-only access (const)
    float operator()(int i, int j) const 
    {
        return data[i * cols + j];
    }

    // Writable access
    float& operator()(int i, int j) 
    {
        return data[i * cols + j];
    }
    
    float& g(int i, int j) 
    {
        return grad[i * cols + j];
    }

    void zero_grad() 
    {
        std::fill(grad.begin(), grad.end(), 0.0f);
    }
};
