# Stanford CS336 Assignment Solutions

This project is my answers for Stanford CS336 Assignment.

## Table of Contents

### [Assignment 1 Chapter 1 Byte-Pair Encoding (BPE) Tokenizer](#assignment-1-chapter-1)
- [Implementing the BPE training](#implementing-the-bpe-training)
- [Implementing the tokenizer](#implementing-the-tokenizer)

### [Assignment 1 Chapter 2 Transformer Language Model Architecture](#assignment-1-chapter-2)
- [Implementing the linear module](#implementing-the-linear-module)
- [Implementing the embedding module](#implementing-the-embedding-module)
- [Implementing the Root Mean Square Layer Normalization (RMSNorm)](#implementing-the-root-mean-square-layer-normalizationrmsnorm)
- [Implementing the position-wise feed-forward network (SwiGLU)](#implementing-the-position-wise-feed-forward-network-swiglu)
- [Implementing the Rotary Position Embedding (RoPE)](#implementing-the-rotary-position-embedding-rope)
- [Implementing the softmax](#implementing-the-softmax)
- [Implementing the scaled dot-product attention](#implementing-the-scaled-dot-product-attention)

---

## Assignment 1 Chapter 1

### Implementing the BPE training

#### Pre-tokenization
- Parallel pre-tokenization can efficiently process large input corpora through streaming

#### BPE Merge
- The key to BPE training is dynamically maintaining byte pairs, tokens, and their respective occurrence counts

#### test 
- implement the test adapter at [adapters.run_train_bpe], and then test using:
```bash
uv run pytest tests/test_train_bpe.py
```

### Implementing the tokenizer
- Do not skip pre-tokenization—applying BPE directly to full sentences corrupts encoding by merging bytes across word boundaries.
- Do not use special tokens in random order—always sort them by length (longest first) to prevent shorter tokens from incorrectly matching parts of longer ones.
- Do not ignore edge cases—always handle inputs like empty strings explicitly to ensure basic functionality passes tests.

#### test 
- implement the test adapter at [adapters.get_tokenizer], and then test using:
```bash
uv run pytest tests/test_tokenizer.py
```
## Assignment 1 Chapter 2

### Implementing the linear module
- Linear weights: truncated at [−3σ, 3σ]
- implement the test adapter at [adapters.run_linear], and then test using:
```bash
uv run pytest -k test_linear
```

### Implementing the embedding module
- inherits from nn.Module
- embedding lookup, embedding matrix shape (vocab_size, d_model) using torch.LongTensor
- embedding weights: truncated at [−3, 3]
- implement the test adapter at [adapters.run_embedding], and then test using:
```bash
uv run pytest -k test_embedding
```

### Implementing the Root Mean Square Layer Normalization(RMSNorm)
- inherits from nn.Module
- using float32 during forwards function to avoid Numerical Overflow or Underflow
- rms_a (batch_size, sequence_length, 1) g_weight (d_model,) 
- x / rms_a * g_weight
- implement the test adapter at [adapters.run_rmsnorm], and then test using:
```bash
uv run pytest -k test_rmsnorm
```


### Implementing the position-wise feed-forward network (SwiGLU)
- SwiGLU composed of a SiLU activation function and a GLU.
- $$\text{SwiGLU}(x) = \text{SiLU}(xW_1) \odot (xW_3)$$
- $$\text{FFN}(x) = \left[ \text{SiLU}(xW_1) \odot (xW_3) \right] W_2$$
- $$\text{SiLU}(x) = x \cdot \sigma(x)$$
- $$\sigma(x) = \frac{1}{1 + e^{-x}}$$
- canonically, d_ff = 8/3 * d_model
- implement the test adapter at [adapters.run_swiglu], and then test using:
```bash
uv run pytest -k test_swiglu
```

### Implementing the Rotary Position Embedding (RoPE)
- use rotary matrix encode position info to Q and K 
- $$x_m' = R_{\Theta,m} x_m$$
- make attn score only relys on relative position (m-n) $$\langle \text{q}_m', \text{k}_n' \rangle = f(\text{q}, \text{k}, m-n)$$
- implement the test adapter at [adapters.run_rope], and then test using:
```bash
uv run pytest -k test_rope
```

### Implementing the softmax
- Subtract max for numerical stability
-  Temperature parameter τ (default=1.0)
-  τ < 1: sharper distribution (amplify differences)
- τ > 1: smoother distribution (reduce differences)
- implement the test adapter at [adapters.run_softmax], and then test using:
```bash
uv run pytest -k test_softmax
```

### Implementing the scaled dot-product attention
- Compute QK^T
- Scale by sqrt(d_k)
- Apply softmax to get attention weights
- Multiply by V
- implement the test adapter at [adapters.run_scaled_dot_product_attention], and then test using:
```bash
uv run pytest -k test_scaled_dot_product_attention
```

