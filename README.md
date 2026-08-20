# mini-GPT: A Transformer Built From Scratch

A GPT-style, decoder-only transformer implemented from scratch in PyTorch and trained on character level Shakespeare no `nn.Transformer`, no shortcuts. Built incrementally, layer by layer, to actually understand what's happening at every step instead of just calling a library.

## Architecture

Standard decoder-only (GPT-style) transformer:

```
tokens → embedding → + positional encoding → [transformer block] × N → LayerNorm → linear → logits
```

Each transformer block:
```
x → multi-head self-attention (causal) → Add & Norm → feedforward → Add & Norm
```

## Project structure

Built incrementally, one component at a time:

| File | Component |
|---|---|
| `self_attention.py` | Single-head scaled dot-product self-attention, built from raw matrix ops |
| `multihead_attention.py` | Multi-head attention with a `causal` toggle (masked/unmasked reusable) |
| `positional_encoding.py` | Sinusoidal positional encoding |
| `ffn_addnorm.py` | Position-wise feedforward network + residual Add & Norm |
| `transformer_block.py` | Full transformer block (attention + FFN + both Add & Norms) |
| `mini_gpt.py` | Full model: embedding → N stacked blocks → final norm → LM head, plus autoregressive `generate()` |
| `data.py` | Character-level tokenizer and batcher (`CharDataset`) |
| `train.py` | Training loop |
| `generate.py` | Load a checkpoint and sample text from a prompt |

A notable design choice: `MultiHeadAttention` takes a `causal` flag rather than hardcoding masking, so the same module could be reused for an encoder later.

## Dataset

Character-level tokenization on the [tiny Shakespeare](https://github.com/karpathy/char-rnn) corpus no tokenizer library, ~65-character vocabulary, easy to reason about end-to-end.

## Model config (as trained)

| Hyperparameter | Value |
|---|---|
| `block_size` (context window) | 128 |
| `d_model` | 192 |
| `n_heads` | 6 |
| `d_ff` | 768 (4 × d_model) |
| `n_layers` | 6 |
| `batch_size` | 64 |
| `max_iters` | 3000 |
| `learning_rate` | 3e-4 |
| Parameters | 2,694,593 |

## Training

<img width="450" height="auto" alt="image" src="https://github.com/user-attachments/assets/6a490291-0e96-44d4-b18a-20e352c39faa" />
