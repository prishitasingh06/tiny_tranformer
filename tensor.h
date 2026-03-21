// tensor.h
#pragma once
#include <vector>

struct Tensor {
    std::vector<float> data;
    int rows, cols;

    Tensor(int r, int c) : rows(r), cols(c), data(r * c) {}

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
};