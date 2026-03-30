#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <string>
#include <iomanip>

#include "tensor.h"
#include "rmsnorm.h"
#include "multi_head_attention.h"
#include "kv_cache.h"
#include "embeddings.h"
#include "linear.h"
#include "ffn.h"
#include "block.h"
#include "gpt.h"
#include "loss.h"
#include "dataset.h"

// global RNG
static std::mt19937 g_rng{std::random_device{}()};

// helper: sample token using softmax + temperature
int sample_token(const Tensor& logits, int vocab_size, float temperature = 0.8f)
{
    float max_val = -1e9f;
    for (int i = 0; i < vocab_size; ++i)
        max_val = std::max(max_val, logits(0,i));

    std::vector<float> probs(vocab_size);
    float sum = 0.f;
    for (int i = 0; i < vocab_size; ++i)
    {
        probs[i] = std::exp((logits(0,i)-max_val)/temperature);
        sum += probs[i];
    }
    for (int i = 0; i < vocab_size; ++i) probs[i] /= sum;

    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return dist(g_rng);
}

int main()
{
    // ---- config ----
    constexpr int vocab_size = 3;
    constexpr int d = 4;
    constexpr int num_layers = 2;
    constexpr int num_heads = 2;
    constexpr int seq_len = 8;

    const std::vector<std::string> vocab = {"pizza", "mushrooms", "pasta"};

    // ---- load embeddings ----
    Tensor E = load_embeddings("tiny_food_embeddings.csv", vocab_size, d);
    std::cout << "Embeddings loaded successfully:\n";
    for (int i = 0; i < vocab_size; ++i)
    {
        std::cout << "Token '" << vocab[i] << "': ";
        for (int j = 0; j < d; ++j)
            std::cout << E(i,j) << " ";
        std::cout << "\n";
    }

    // ---- create GPT model ----
    GPT model(vocab_size, d, num_layers, num_heads);

    // copy embeddings
    for (int i = 0; i < vocab_size; ++i)
        for (int j = 0; j < d; ++j)
            model.embeddings(i,j) = E(i,j);

    // ---- autoregressive generation ----
    int current_token = 1; // seed token = mushrooms
    std::cout << "\nAutoregressive generation using GPT struct:\n";
    std::cout << "Seed token: " << vocab[current_token] << "\n\n";

    for (int step = 0; step < seq_len; ++step)
    {
        Tensor logits = model.forward(current_token);
        int next_token = 0;
        float max_val = logits(0,0);
        for (int v = 1; v < vocab_size; ++v)
        {
            if (logits(0,v) > max_val)
            {
                max_val = logits(0,v);
                next_token = v;
            }
        }

        std::cout << "Step " << step
                  << " token: " << std::left << std::setw(12) << vocab[next_token]
                  << " logits:";
        for (int v = 0; v < vocab_size; ++v)
            std::cout << " " << std::fixed << std::setprecision(4) << logits(0,v);
        std::cout << "\n";

        current_token = next_token;
    }

    return 0;
}