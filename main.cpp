
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <string>
#include "tensor.h"
#include "rmsnorm.h"
#include "multi_head_attention.h"
#include "kv_cache.h"
#include "embeddings.h"
#include "linear.h"

// global
static std::mt19937 g_rng{std::random_device{}()};

//my helper funcs

// Safe lm-head forward: logits[vocab] = W[vocab x d] · x[d]
// have replaced Linear::forward to make dimension errors visible at compile time.
inline Tensor lm_head_forward(const Linear& lm_head,
                               const Tensor& x,
                               int vocab_size, int d)
{
    Tensor logits(1, vocab_size);
    for (int v = 0; v < vocab_size; ++v)
    {
        float dot = 0.f;
        for (int k = 0; k < d; ++k)
            dot += lm_head.W(v, k) * x(0, k);
        logits(0, v) = dot;
    }
    return logits;
}

// Softmax sampling with temperature (temperature < 1 → sharper picks)
int sample_token(const Tensor& logits, int vocab_size, float temperature = 0.8f)
{
    float max_val = -1e9f;
    for (int i = 0; i < vocab_size; ++i)
        max_val = std::max(max_val, logits(0, i));

    std::vector<float> probs(vocab_size);
    float sum = 0.f;
    for (int i = 0; i < vocab_size; ++i)
    {
        probs[i] = std::exp((logits(0, i) - max_val) / temperature);
        sum += probs[i];
    }
    for (int i = 0; i < vocab_size; ++i) probs[i] /= sum;

    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return dist(g_rng);
}

/**
 * main
*/
int main()
{
    constexpr int vocab_size  = 3;
    constexpr int d           = 4;
    constexpr int num_heads   = 2;
    constexpr int seq_len     = 8;
    constexpr int max_seq_len = 16;

    const std::vector<std::string> vocab = {"pizza", "mushrooms", "pasta"};


/**
 * embeddings
*/
Tensor E = load_embeddings("tiny_food_embeddings.csv", vocab_size, d);

// debugging: to check if CSV loaded correctly
bool csv_loaded = true;
for (int i = 0; i < vocab_size; ++i) 
{
    for (int j = 0; j < d; ++j) {
        if (std::isnan(E(i, j))) 
        {
            csv_loaded = false;
            break;
        }
    }
}
if (csv_loaded) 
{
    std::cout << "Embeddings loaded successfully:\n";
    for (int i = 0; i < vocab_size; ++i) 
    {
        std::cout << "Token '" << vocab[i] << "': ";
        for (int j = 0; j < d; ++j)
            std::cout << E(i, j) << " ";
        std::cout << "\n";
    }
} 
else 
{
    std::cerr << "Error: embeddings contain NaNs or were not loaded correctly!\n";
}
    // ── LM head ──────────────────────────────────────────────────────────
    Linear lm_head(vocab_size, d);
    {
        std::uniform_real_distribution<float> uni(-0.5f, 0.5f);
        for (int i = 0; i < vocab_size; ++i)
            for (int j = 0; j < d; ++j)
                lm_head.W(i, j) = uni(g_rng);
    }

    //initial token
    std::uniform_int_distribution<int> tok_dist(0, vocab_size - 1);
    int current_token = tok_dist(g_rng);

    Tensor layer_input(1, d);
    for (int j = 0; j < d; ++j)
        layer_input(0, j) = E(current_token, j);

    KVCache cache(max_seq_len, d);

    std::cout << "Autoregressive generation (temperature-sampled softmax):\n";
    std::cout << "Seed token: " << vocab[current_token] << "\n\n";

    for (int step = 0; step < seq_len; ++step) {

        // single-layer attention with KV cache
        Tensor Q     = layer_input;        // query  = current embedding
        Tensor K_new = layer_input;        // key    = same
        Tensor V_new = layer_input;        // value  = same

        Tensor attn_out = multi_head_attention_kv(Q, K_new, V_new, cache, num_heads);

/* Residual connection
-> Residual keeps the original embedding signal alive, preventing attention from collapsing the representation into near-zero noise.

-> We add the input embedding x back to the attention output:  y = Attn(x) + x

-> Why this is important?
In this tiny untrained model, attention is effectively a random transformation. 
That means Attn(x) can be nearly orthogonal to x:  Attn(x) ⟂ x  ⇒  Attn(x) · x ≈ 0

The LM head computes logits via dot products: logits[v] = W[v] · y

-> If we omit the residual:
    y = Attn(x)
    ⇒ logits[v] = W[v] · Attn(x) ≈ 0   for all v

This happens because Attn(x) has little alignment with the learned directions in W, so all logits become very small and similar:
    logits ≈ [ε, ε, ε]  (ε ≈ 0)

Softmax then becomes nearly uniform:
    softmax(logits) ≈ [1/V, 1/V, ..., 1/V]
→ model loses all preference between tokens.

With the residual: y = Attn(x) + x ≈ x + small_noise

Now logits preserve meaningful signal: logits[v] = W[v] · x + small perturbation
→ differences between tokens survive, and sampling works.
*/
        for (int j = 0; j < d; ++j)
            attn_out(0, j) += layer_input(0, j);

        // RMSNorm with safe epsilon
        // rmsnorm implementation uses eps ≥ 1e-6 internally.
        rmsnorm(attn_out);  // modifies in-place

        Tensor logits = lm_head_forward(lm_head, attn_out, vocab_size, d);

        int next_token = sample_token(logits, vocab_size);

        // debugging
        std::cout << "Step " << step
                  << "  token: " << std::left << std::setw(12) << vocab[next_token]
                  << "  logits:";
        for (int j = 0; j < vocab_size; ++j)
            std::cout << "  " << std::fixed << std::setprecision(4) << logits(0, j);
        std::cout << "\n";

        // adv
        if (step + 1 < seq_len)
            for (int j = 0; j < d; ++j)
                layer_input(0, j) = E(next_token, j);

        current_token = next_token;
    }

    return 0;
}