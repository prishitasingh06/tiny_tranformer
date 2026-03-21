My attempt at building a transformer from scratch
***
# Tiny Transformer

This project is very close to me. I didn’t build this just to “have a transformer implementation” ,I built it to truly understand what’s happening under the hood. Every line of code represents something I once found confusing, abstract, or hidden behind high-level libraries. Writing it all from scratch in C++ forced me to slow down and actually *learn*.
This is my attempt to **understand how transformers actually work** not just by using libraries, but by building one from scratch in C++.
Instead of relying on frameworks, I implemented the core pieces myself: embeddings, attention, normalization, and autoregressive generation. A lot of the math and intuition came from courses and resources I studied, but the real understanding came from implementing it myself in C++.


Modern machine learning workflows rely heavily on frameworks. While that is extremely powerful, at times I did not understand them comepletely. I wanted to change that.
So I set out to:

* Break down how transformers actually work
* Rebuild the core components from first principles
* Translate theory into something concrete and executable

This project is the result of that process.


Math & logic behind it in simple words-
* **Attention** is just weighted relationships between tokens
* **Embeddings** turn words into meaningful numeric vectors
* **Logits → probabilities** is simply normalization via softmax
* **Generation** is a loop: predict → append → repeat

--
## Overview
A minimal transformer inference engine implementing:
- RoPE positional encoding
- RMSNorm
- Multi-head attention
- Causal masking
- KV caching for autoregressive generation

Designed to run **entirely on Mac**, no PyTorch required.

## Features
- Forward pass only (no training till now)
- Random or real embeddings
- Modular design for experimentation
- Ready for optimization (SIMD / NEON)

## Usage
1. Build & run:
```bash
make run
./run   


## What I Implemented

Everything here is built from scratch,no ML libraries and no shortcuts:

* Matrix operations (the backbone of everything)
* Token embeddings
* Scaled dot-product attention
* Softmax and probability distributions
* Basic normalization
* Autoregressive text generation



##  What I Learned

This project helped me connect concepts that previously felt disconnected:
Things that once felt like “magic” now feel mechanical and understandable.

## Tokenization & Embeddings

To keep things simple and transparent, I created a small custom embedding dataset in CSV format:

```
token,e1,e2,e3,e4
pizza,0.25,0.10,-0.15,0.40
mushrooms,-0.12,0.30,0.22,-0.05
pasta,0.05,-0.20,0.35,0.15
```

This helped me clearly see:

* How tokens map to vectors
* How similarity emerges from numbers
* How these vectors flow through attention


## Please note-
This project isn’t about performance or scale. It is about clarity.
It represents the moment where transformers stopped being something I used and became something I actually understand*
