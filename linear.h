#pragma once
#include "tensor.h"

struct Linear {
    Tensor W; // weights: vocab_size x d
    Linear(int vocab_size, int d) : W(vocab_size,d) {}

    // simple forward: output logits = W * x^T
    Tensor forward(Tensor& x){
        int n=x.rows;
        int d=x.cols;
        int vocab_size=W.rows;
        Tensor logits(n,vocab_size);
        for(int i=0;i<n;i++){
            for(int v=0;v<vocab_size;v++){
                float sum=0;
                for(int j=0;j<d;j++) sum += x(i,j)*W(v,j);
                logits(i,v)=sum;
            }
        }
        return logits;
    }
};