# Character-level GPT from scratch in NumPy

**Foundations of Machine Learning — Final Project, Track 3 (GPT Report)**

A decoder-only Transformer that generates text one character at a time, implemented **entirely
from scratch in NumPy** — every layer, every backward pass, the optimizer, and the tokenizer. No
PyTorch, no TensorFlow, no autograd.

Trained on **Tiny Shakespeare** (1.1 M characters), it reaches **1.81 nats/char** on held-out
validation data — perplexity **6.11**, or **11× better than random guessing**.

```
ROMEO:
What Duke in spy treat and some do idle thee,
Let sance, you see may thou would with peindon,
Than you what oples the tant the citions with plous,
```
*(generated, temperature 0.8, from the prompt `ROMEO:`)*

---

## What is implemented from scratch

| Component | What is hand-derived |
|---|---|
| `Linear` | forward + `d_w`, `d_b`, `d_x`; Xavier-style init |
| `Embedding` | row lookup + **scatter-add** backward (repeated ids accumulate) |
| `CausalSelfAttention` | Q/K/V, scaled scores, causal mask, and the **full backward chain** through the softmax Jacobian |
| `MultiHeadAttention` | parallel heads, concat, output projection, per-head gradient split |
| `LayerNorm` | forward + the **three-term** input gradient (mean and variance corrections) |
| `Gelu` | tanh approximation + its derivative |
| `FeedForward` | `Linear → GELU → Linear` with ×4 expansion |
| `TransformerBlock` | pre-norm ordering + residual connections |
| `CharGPT` | token + positional embeddings, N blocks, final norm, LM head, autoregressive `generate()` |
| `cross_entropy` | softmax cross-entropy with the combined `softmax − onehot` gradient |
| `Adam` | moments, bias correction, in-place updates |
| `CharTokenizer` | character vocabulary with `<bos>` / `<eos>` / `<pad>` |

Every backward pass is verified against **finite differences** (`numeric_gradient`). The notebook
runs **65 assertions**, all passing.

## Repository layout

```
notebooks/
  Implementation.ipynb    the deliverable: components, tests, training, results
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
| Parameters | 620,740 (`d_model=128`, 4 heads, 3 blocks, 64-char context) |
| Validation loss | **1.810** nats/char (uniform baseline 4.220) |
| Validation perplexity | **6.11** (baseline 68) |
| Train/val gap | 0.125 nats — generalizing, not memorizing |

One finding worth reporting:

**Initialization is not cosmetic.** With `Linear` initialized as plain `standard_normal` (no
`1/√n_in`), the model started at loss ≈14.8 instead of `ln(vocab)` ≈3.4, stalled near 2.5, and
produced gibberish. Scaling the init fixed it — same architecture, same steps, loss 1.81 and
readable text. **Every gradient check passed while it was broken**: the gradients were correct,
the *scale* was wrong, which is exactly the class of bug tests do not catch.

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
