#include <iostream>
#include "gpt.h"
#include "loss.h"
#include "dataset.h"

// Example training loop for a tiny GPT model
int main()
{
    // Step 1: Initialize GPT model
    // Arguments: vocab_size=3, d_model=4, num_layers=2, num_heads=2
    GPT model(3, 4, 2, 2);

    // Step 2: Load dataset
    // get_data() returns a vector<int> of token IDs
    auto data = get_data();

    // Step 3: Training loop
    for (int epoch = 0; epoch < 200; ++epoch)
    {
        float total_loss = 0;  // Accumulate loss over this epoch

        // Step 3a: Iterate through tokens
        for (int i = 0; i < data.size()-1; ++i)
        {
            int input = data[i];      // Current token
            int target = data[i+1];   // Next token to predict

            // Step 3b: Forward pass through GPT
            Tensor logits = model.forward(input);

            // Step 3c: Compute cross-entropy loss
            float loss = cross_entropy(logits, target);

            // Step 3d: Accumulate loss
            total_loss += loss;
        }

        // Step 3e: Print epoch and total loss
        std::cout << "Epoch " << epoch << " Loss: " << total_loss << "\n";
    }
    return 0;
}

/* 
Note for anyone reading this file-
I have not implemented backpropagation yet
This is just has forward pass + loss computation
*/