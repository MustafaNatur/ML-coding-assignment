# Character-level GPT from scratch in NumPy

**Foundations of Machine Learning — Final Project, Track 3 (GPT Report)**

A decoder-only Transformer that generates text one character at a time, implemented **entirely
from scratch in NumPy** — every layer, every backward pass, the optimizer, and the tokenizer. No
PyTorch, no TensorFlow, no autograd.

Trained on **Tiny Shakespeare** (1.1 M characters), it reaches **1.82 nats/char** on held-out
validation data — perplexity **6.18**, or **10.5× better than random guessing**.

```
ROMEO:
What Duke in ply pure,
And so, I say them the now see compled be may sit?
To make come made quiciague, then now way, is roping down homs,
```
*(generated, temperature 0.8, from the prompt `ROMEO:`)*

---

## What is implemented from scratch

| Component | What is hand-derived |
|---|---|
| `Layer` / `Sequential` | the shared interface: parameter collection and gradient clearing, derived once from `parameter_names` and `children()` |
| `Linear` | forward + `d_W`, `d_b`, `d_x`; `1/√n_in` init |
| `Embedding` | row lookup + **scatter-add** backward (repeated ids accumulate) |
| `CausalSelfAttention` | Q/K/V, scaled scores, causal mask, and the **full backward chain** through the softmax Jacobian |
| `MultiHeadAttention` | parallel heads, concat, output projection, per-head gradient split |
| `LayerNorm` | forward + the **three-term** input gradient (mean and variance corrections) |
| `Gelu` | tanh approximation + its derivative |
| `FeedForward` | `Linear → GELU → Linear` with ×4 expansion, as a `Sequential` |
| `ResidualSublayer` | the pre-norm residual pattern `X + branch(norm(X))`, written once and used twice per block |
| `TransformerBlock` | attention sub-layer, then feed-forward sub-layer |
| `CharGPT` | token + positional embeddings, N blocks, output head, autoregressive `generate()` |
| `cross_entropy` | softmax cross-entropy with the fused `softmax − onehot` gradient |
| `Adam` | moments, bias correction, in-place updates |
| `CharTokenizer` | character vocabulary, text ↔ ids |

Every backward pass was derived by hand and checked against **finite differences**
(`numeric_gradient`). The notebook keeps one **Demonstration** cell that gradient-checks the fully
assembled model end to end — relative error around `1e-07` through the character embedding table
and below `1e-09` through an attention query weight — and shows that causal attention leaks exactly
`0.00e+00` from future tokens.

## Repository layout

```
notebooks/
  Implementation.ipynb    the deliverable: components, demonstration, training, results
  Theory.ipynb            theory write-up behind each component
  data/                   Tiny Shakespeare (auto-downloaded if absent)
IMPLEMENTATION.md         build spec: architecture, component contracts, results
LEARNING.md               study plan — what to learn for each component
BACKLOG.md                components to re-implement solo later, with original papers
```

## Running it

Open `notebooks/Implementation.ipynb` in Jupyter and run all cells (~3 minutes; most of it is the
16,000-step training run).

Requirements: **NumPy** for the model, **Matplotlib** for the loss-curve and hyperparameter plots.

Using the API directly:

```python
tokenizer = CharTokenizer(text)
model = CharGPT(tokenizer.vocab_size, d_model=128, number_of_heads=4,
                number_of_blocks=3, max_sequence_length=64)

train_with_validation(model, train_token_ids, validation_token_ids,
                      steps=16000, block_size=64, lr=5e-4)

print(model.generate(tokenizer, max_new_tokens=400, temperature=0.8, prompt="ROMEO:"))
```

## Results

| | Value |
|---|---|
| Parameters | 619,969 (`d_model=128`, 4 heads, 3 blocks, 64-char context) |
| Validation loss | **1.821** nats/char (uniform baseline 4.174) |
| Validation perplexity | **6.18** (baseline 65) |
| Train/val gap | 0.127 nats — generalizing, not memorizing |

Two findings worth reporting:

**Initialization is not cosmetic.** With `Linear` initialized as plain `standard_normal` (no
`1/√n_in`), the model started at loss ≈14.8 instead of `ln(vocab)` ≈4.2, stalled near 2.5, and
produced gibberish. Scaling the init fixed it — same architecture, same steps, loss 1.82 and
readable text. **Every gradient check passed while it was broken**: the gradients were correct,
the *scale* was wrong, which is exactly the class of bug tests do not catch.

**A short sweep ranks early learning speed, not final quality.** In the 2,000-step hyperparameter
study the 1-block model wins (2.336 vs the 3-block baseline's 2.365), because smaller models
converge faster early; over the full budget the 3-block model reaches 1.82. The learning rate
dominates everything else — 10× too high costs ~0.3 nats, far more than any architectural change
tested.

Note that results move slightly between runs despite fixed seeds: floating-point matrix products
are not associative, and over 16,000 steps that is enough to change generated text entirely. Quote
these numbers to two decimals.

## Known limitations

- **No batching.** Every layer takes a single sequence `(T, d_model)`. That kept the attention and
  LayerNorm backward passes directly gradient-checkable, at the cost of sample efficiency — one
  window per optimizer step. Adding a leading batch axis is the obvious next optimization.
- **Character-level tokenizer.** Simple and fully from scratch, but it makes sequences long. BPE
  would shorten them; it is tracked in [BACKLOG.md](BACKLOG.md).
- **Model size.** At ~0.6 M parameters, samples are locally fluent but not semantically coherent —
  expected, and explicitly anticipated for Track 3.

## Scope

This branch is the **text GPT only**, which is a complete Track-3 project. The multimodal
image-captioning extension — patch embeddings, a split mask (image tokens unmasked, caption tokens
causal), Flickr8k, and an attention-based grounding analysis — is planned separately on `main`.
