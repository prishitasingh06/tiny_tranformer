#pragma once
#include "tensor.h"
#include "block.h"
#include "linear.h"
#include <vector>

//GPT model (decoder-only transformer)
struct GPT 
{
    int vocab_size;       // Number of tokens in vocabulary
    int d;                // Model dimension (embedding size)
    int num_layers;       // Number of transformer blocks
    int num_heads;        // Number of attention heads per block

    Tensor embeddings;                    // Token embeddings
    std::vector<TransformerBlock> layers; // Transformer blocks
    std::vector<KVCache> caches;          // KV caches for each layer (autoregressive decoding)
    Linear lm_head;                       // Linear layer to produce logits

    // Constructor: initializing model parameters and layers
    GPT(int vocab, int d_model, int layers_n, int heads)
        : vocab_size(vocab),
          d(d_model),
          num_layers(layers_n),
          num_heads(heads),
          embeddings(vocab, d_model),
          lm_head(vocab, d_model)
    {
        for (int i = 0; i < num_layers; ++i)
        {
            // Add a transformer block
            layers.emplace_back(d, num_heads);
            // Initialize KV cache for this layer--> max sequence length 32
            caches.emplace_back(32, d);
        }
    }

    // Embed a single token into its vector representation
    Tensor embed(int token)
    {
        Tensor x(1, d);
        for (int j = 0; j < d; ++j)
        {
            x(0,j) = embeddings(token,j);
        }
        return x;
    }

    // Forward pass for a single token
    Tensor forward(int token)
    {
        std::cout << "[DEBUG] GPT forward called with token=" << token << "\n";
        // Step 1: Convert token to embedding
        Tensor x = embed(token);

        // Step 2: Pass through all transformer layers
        for (int i = 0; i < num_layers; ++i)
        {
            x = layers[i].forward(x, caches[i]);
        }

        // Step 3: Compute logits for vocabulary
        Tensor logits(1, vocab_size);
        for (int v = 0; v < vocab_size; ++v)
        {
            float val = 0;

            for (int j = 0; j < d; ++j)
            {
                // Explicit if/else instead of ternary
                if (lm_head.W(v,j) != 0)
                {
                    val += lm_head.W(v,j) * x(0,j);
                }
                else
                {
                    val += 0;
                }
            }
            logits(0,v) = val;
        }
        return logits;
    }
};