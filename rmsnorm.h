#pragma once
#include "tensor.h"
#include <cmath>

void rmsnorm(Tensor& x,float eps=1e-6)
{
    for(int i=0;i<x.rows;i++)
    {
        float mean_sq=0;
        for(int j=0;j<x.cols;j++) mean_sq += x(i,j)*x(i,j);
        mean_sq /= x.cols;
        float scale = 1.0f/std::sqrt(mean_sq+eps);
        for(int j=0;j<x.cols;j++) x(i,j)*=scale;
    }
}