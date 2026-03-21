#pragma once
#include "tensor.h"
#include "rope.h"
#include "kv_cache.h"
#include <cmath>
#include <algorithm>

Tensor multi_head_attention_kv(Tensor& Q, Tensor& K_new, Tensor& V_new, KVCache& cache, int num_heads){
    int n=Q.rows, d=Q.cols, head_dim=d/num_heads;

    cache.append(K_new,V_new);

    Tensor out(n,d);

    for(int h=0;h<num_heads;h++){
        Tensor Qh(n,head_dim), Kh(cache.seq_len,head_dim), Vh(cache.seq_len,head_dim);

        for(int i=0;i<n;i++) for(int j=0;j<head_dim;j++) Qh(i,j)=Q(i,j+h*head_dim);
        for(int i=0;i<cache.seq_len;i++) for(int j=0;j<head_dim;j++){
            Kh(i,j)=cache.K(i,j+h*head_dim);
            Vh(i,j)=cache.V(i,j+h*head_dim);
        }

        apply_rope(Qh); apply_rope(Kh);

        Tensor scores(n,cache.seq_len);
        for(int i=0;i<n;i++){
            for(int j=0;j<cache.seq_len;j++){
                if(j>i+cache.seq_len-n) scores(i,j)=-1e9;
                else{
                    float dot=0;
                    for(int k=0;k<head_dim;k++) dot+=Qh(i,k)*Kh(j,k);
                    scores(i,j)=dot/std::sqrt(head_dim);
                }
            }
            float max_val=-1e9;
            for(int j=0;j<cache.seq_len;j++) max_val=std::max(max_val,scores(i,j));
            float sum=0;
            for(int j=0;j<cache.seq_len;j++){ scores(i,j)=std::exp(scores(i,j)-max_val); sum+=scores(i,j); }
            for(int j=0;j<cache.seq_len;j++) scores(i,j)/=sum;
        }

        Tensor head_out(n,head_dim);
        for(int i=0;i<n;i++){
            for(int k=0;k<head_dim;k++){
                float val=0;
                for(int j=0;j<cache.seq_len;j++) val+=scores(i,j)*Vh(j,k);
                head_out(i,k)=val;
            }
        }

        for(int i=0;i<n;i++) for(int j=0;j<head_dim;j++) out(i,j+h*head_dim)=head_out(i,j);
    }

    return out;
}