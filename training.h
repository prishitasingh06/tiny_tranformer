#include <iostream>
#include "gpt.h"
#include "loss.h"
#include "dataset.h"
#include "backward.h"   //contains lm_head_backward + update_weights

// Example training loop for a tiny GPT model
int main()
{
    // Step 1: Initialize GPT model
    // Arguments: vocab_size=3, d_model=4, num_layers=2, num_heads=2
    GPT model(3, 4, 2, 2);

    // Step 2: Load dataset
    // get_data() returns a vector<int> of token IDs
    auto data = get_data();

    // Learning rate for SGD
    float lr = 0.01f;

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
            Tensor x = model.embed(input);   // keep input embedding--> needed for backprop
            Tensor logits = model.forward(input);

            // Step 3c: Compute cross-entropy loss + gradients
            Tensor d_logits(1, model.vocab_size);  // gradient wrt logits
            float loss = cross_entropy_with_grad(logits, target, d_logits);

            // Step 3d: Accumulate loss
            total_loss += loss;

            // Step 3e: Backward pass (LM head only for now)
            Tensor d_x(1, model.d);  // gradient flowing backward into representation
            lm_head_backward(x, model.lm_head, d_logits, d_x);

            // Step 3f: Update weights (SGD)
            update_weights(model.lm_head.W, lr);
        }

        // Step 3g: Print epoch and total loss
        std::cout << "Epoch " << epoch << " Loss: " << total_loss << "\n";
    }
    return 0;
}

/* 
Note for anyone reading this file-
Current implementation has- 
- fwd pass + loss computation
- backpropagation (LM head only)
- gradient computation via function cross_entropy_with_grad
- weight updates using SGD

Limitations- 
- Attention weights are not trained yet
- FFN weights are not trained yet
*/
