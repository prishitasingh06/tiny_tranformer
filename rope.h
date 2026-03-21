#pragma once
#include "tensor.h"
#include <cmath>

void apply_rope(Tensor& x)
{
    int n=x.rows, d=x.cols;
    for(int pos=0;pos<n;pos++)
    {
        for(int i=0;i<d;i+=2)
        {
            float theta = std::pow(10000.0f, - (float)i/d);
            float angle = pos*theta;
            float x1 = x(pos,i), x2 = x(pos,i+1);
            x(pos,i)   = x1*std::cos(angle) - x2*std::sin(angle);
            x(pos,i+1) = x1*std::sin(angle) + x2*std::cos(angle);
        }
    }
}