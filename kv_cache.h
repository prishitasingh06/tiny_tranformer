#pragma once
#include "tensor.h"

struct KVCache {
    Tensor K;
    Tensor V;
    int seq_len;

    KVCache(int max_seq_len,int d) : K(max_seq_len,d), V(max_seq_len,d), seq_len(0) {}

    void append(Tensor& newK, Tensor& newV){
        for(int i=0;i<newK.rows;i++) for(int j=0;j<newK.cols;j++){
            K(seq_len+i,j)=newK(i,j);
            V(seq_len+i,j)=newV(i,j);
        }
        seq_len += newK.rows;
    }
};