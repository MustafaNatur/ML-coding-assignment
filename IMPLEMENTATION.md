# Implementation Plan — Image Captioning Transformer from Scratch

The **build spec**. For the theory behind each step, see [LEARNING.md](LEARNING.md)
(same step numbers).

> **Golden rule:** anything with a backward pass isn't done until a **finite-difference
> gradient check passes** (built in Step 0).

---

## Task

Build a **decoder-only Transformer that generates a text caption for an image**, entirely
from scratch in NumPy (Track 3). Given a new image, the model outputs a sentence describing
it, one token at a time. Captioning = next-token language modeling conditioned on an image.

- **Dataset:** Flickr8k (real photos + human captions), downscaled, ~1–2k-image subset.
- **Deliverable:** a Jupyter notebook that runs start→finish + a ~6-page report.
- **Constraint:** network from scratch; only NumPy for the model (sklearn/Matplotlib for
  prep/plots, tokenizer may be self-written).

---

## Architecture choice — decoder-only with an image prefix

The classic captioning design is **encoder-decoder**: an image encoder produces K/V, a text
decoder reads them via **cross-attention**. We chose the other valid design:

```
Encoder-decoder :  image → encoder → K,V → text decoder w/ CROSS-attention → caption
Ours            :  [image tokens | caption tokens] → ONE decoder stack → caption
```

**Why:** self-attention over the concatenated sequence already does the cross-modal mixing —
a caption token attending over the prefix *is* attending to the image. So we implement **one**
attention class instead of two (no separate cross-attention to hand-backprop), it matches how
modern multimodal LLMs condition on images, and it stays exactly Track 3 (decoder-only,
next-token prediction).

**Consequence — a split mask:** image tokens are **unmasked** (context), caption tokens are
**causal**. So `CausalSelfAttention` (currently fully causal for the char-GPT) needs an
**optional mask argument** at the vision-bridge step.

---

## Architecture overview

This is the **actual working process** — the forward data-flow the model runs to turn an
image + caption tokens into next-token predictions. It's shared by both modes and only
differs at the very end: **training** feeds the predictions into cross-entropy; **generation**
samples the next token and loops. (The optimizer/learning view is the class diagram below.)

Shapes use `d_model=128`, 64×64 image, 8×8 patches → 64 image tokens.

```mermaid
flowchart TD
    IMG["IMAGE (64,64,3)"] --> PATCH["Patchify"]
    PATCH -->|"(64, 192)"| PEMB["Patch Embedding"]

    CAP["CAPTION 'a dog runs'"] --> TOK["Tokenizer"]
    TOK -->|"ids (T_text,)"| TEMB["Token Embedding"]

    PEMB -->|"(64, 128)"| CAT["Concatenate"]
    TEMB -->|"(T_text, 128)"| CAT

    CAT -->|"(64 + T_text, 128)"| POS["+ Positional Embedding"]
    POS --> BLK["N x Transformer Block<br/>LayerNorm, Attention, +residual<br/>LayerNorm, MLP, +residual"]
    BLK -->|"(64 + T_text, 128)"| FLN["Final LayerNorm"]
    FLN --> OUT["Output Linear"]
    OUT -->|"(T_text, vocab) - caption positions only"| SM["Softmax"]

    SM --> CE["Cross-Entropy<br/>(training)"]
    SM --> GEN["Sample next token<br/>(generation)"]

    classDef img fill:#e3f2fd,stroke:#1565c0;
    classDef txt fill:#f1f8e9,stroke:#558b2f;
    classDef out fill:#fff3e0,stroke:#e65100;
    class IMG,PATCH,PEMB img;
    class CAP,TOK,TEMB txt;
    class CE,GEN out;
```

## Components & connections

| Component | Responsibility | In → Out | Connects |
|---|---|---|---|
| **Image preprocess** *(offline)* | decode + resize to fixed size, save as array | JPEG → `(64,64,3)` | feeds Patchify |
| **Patchify** | cut image into non-overlapping patches, flatten each | `(64,64,3)` → `(64, 192)` | → Patch Embedding |
| **Patch Embedding** | Linear projecting each patch to model width → "image tokens" | `(64, 192)` → `(64, 128)` | → Concatenate |
| **Tokenizer** | caption text ↔ integer ids; adds `<bos>`/`<eos>` | text → `(T_text,)` | → Token Embedding |
| **Token Embedding** | lookup table: id → learned vector | `(T_text,)` → `(T_text,128)` | → Concatenate |
| **Concatenate** | build one sequence: image tokens then caption tokens | → `(64+T_text, 128)` | → Positional |
| **Positional Embedding** | add position info (attention is order-blind without it) | same shape | → Blocks |
| **LayerNorm** | normalize features per token; stabilizes training | `(T,128)` → `(T,128)` | inside each block (×2) + final |
| **Multi-Head Attention** | mix info across positions; **image unmasked, caption causal** | `(T,128)` → `(T,128)` | core of each block |
| **Residual** | add sublayer input back to output; keeps gradients flowing | same shape | wraps attention + MLP |
| **MLP** (Linear→GELU→Linear) | per-token nonlinear transform (×4 width expand) | `(T,128)` → `(T,128)` | second sublayer of block |
| **Transformer Block ×N** | one round of "attend, then think"; stacked for depth | `(T,128)` → `(T,128)` | chained N times |
| **Final LayerNorm** | normalize before the output projection | `(T,128)` → `(T,128)` | → Output Linear |
| **Output Linear** (LM head) | project each token to vocabulary-sized logits | `(T,128)` → `(T,vocab)` | → Softmax |
| **Softmax** | logits → probability distribution over next token | `(T,vocab)` | → loss / sampling |
| **Cross-Entropy** *(train)* | error vs. true next token, **caption positions only** | → scalar loss | drives backprop |
| **Generation** *(inference)* | sample next token, append, repeat until `<eos>` | → caption text | the demo |

**Key connections:**
- **Two modalities, one sequence** — patch and token embeddings both output `d_model`
  vectors, so attention mixes image + text in the same space.
- **Split mask** — image tokens see everything; caption tokens are causal.
- **Loss ignores image positions** — only caption tokens have a "correct next token."
- **One block = attend, then process** — attention shares info between tokens; MLP transforms
  each token; residual + LayerNorm keep the stack trainable.

## Input / Output

- **Training input:** `(image (64,64,3), caption ids (T_text,))` pairs.
- **Training output:** scalar cross-entropy loss on caption positions → gradients.
- **Inference input:** one image `(64,64,3)`.
- **Inference output:** a caption string, generated token by token until `<eos>`.

**Runtime (inference) flow** — image encoded once, caption built autoregressively:
```
encode image ONCE → 64 image tokens
start: [image tokens | <bos>]
loop until <eos>/max-len:
   forward pass → take probs at LAST position → pick token → append
```
| Iter | Model input | Predicts | Caption |
|---|---|---|---|
| 1 | `[img] <bos>` | `a` | a |
| 2 | `[img] <bos> a` | `dog` | a dog |
| 3 | `[img] <bos> a dog` | `runs` | a dog runs |
| 4 | `[img] <bos> a dog runs` | `<eos>` | **done** |

## Software design (components & responsibilities)

Four kinds of component, each with one responsibility:
**layers compute**, the **loss** scores + seeds backprop, the **optimizer** updates,
the **tokenizer** converts text ↔ ids. Layers share one interface so the optimizer treats
them all the same.

### Layer (interface)
```
Layer   (Linear, Embedding, Attention, ...)   →  COMPUTE
    - holds params (values) + grads (filled by backward)
    - forward(), backward()
    - parameters() -> [(value, grad), ...]   # uniform interface for the optimizer
    - zero_grad()
```

### Linear  — fully-connected layer  `Y = X @ W + b`
```
    - params: W (n_in, n_out), b (n_out)   ; grads: d_w, d_b
    - forward(X)   -> X @ W + b            ; caches X
    - backward(d_y) -> d_x                   ; fills d_w = X.T @ d_y, d_b = sum(d_y)
    - parameters() -> [(W, d_w), (b, d_b)]
```

### Embedding  — token id → learned vector
```
    - param: table (vocab_size, d)          the ONE parameter (no bias) ; grad: d_table
    - forward(ids)   -> table[ids]         row lookup ; caches ids
    - backward(d_out) -> None               scatter-add into d_table (np.add.at)
    - parameters()   -> [(table, d_table)]
```
Differs from Linear in two ways: backward is **scatter-add** (duplicate ids accumulate),
and it returns **no input gradient** (`None`) — integer ids aren't differentiable and it's
always the first layer. Same `parameters()` contract, so the optimizer needs no special case.

### CausalSelfAttention  — mixing information across positions
```
    - params: three Linear projections (query, key, value)  -> Q, K, V
    - forward(X)    -> softmax(Q @ K.T / sqrt(d_head) + causal_mask) @ V
                       X (T, d_model) -> output (T, d_head) ; caches Q, K, V, weights
    - backward(d_out) -> d_x   (T, d_model)
    - parameters()  -> q + k + v projection params
```
Backward is the hardest chain in the project — four links: the weighted sum `weights @ V`,
the row-wise **softmax Jacobian** `∂L/∂s = p * (∂L/∂p − Σ(∂L/∂p·p))`, the `1/√d_head` scaling,
then `Q Kᵀ`. Because X feeds all three projections, `d_x` is the **sum** of the three paths.
Masked entries have `p = 0`, so no gradient leaks to the future.

### MultiHeadAttention  — several attention patterns at once
```
    - h heads (each d_head = d_model / h) + one output_projection Linear(d_model, d_model)
    - forward(X)      -> concat(head outputs) -> output_projection   (T, d_model)
    - backward(d_out) -> split the gradient per head, SUM their d_x
    - parameters()    -> all heads' params + output_projection's
```
Rejects a `d_model` not divisible by `number_of_heads` rather than misbehaving quietly.

### LayerNorm  — keeping activations at a stable scale
```
    - params: gamma (d_model,) scale [init 1], beta (d_model,) shift [init 0]
              grads: d_gamma, d_beta
    - forward(X)   -> gamma * (X - mean) / sqrt(var + eps) + beta
                      mean/var over the LAST axis (one token), not over the batch
                      caches normalized_input (x_hat) and standard_deviation
    - backward(d_y) -> d_x   (..., d_model)
    - parameters() -> [(gamma, d_gamma), (beta, d_beta)]
```
Backward is the second-hardest derivation after attention: `mu` and `sigma` depend on **every**
feature of a token, so one feature moves all of that token's outputs. Hence three terms — the
direct path, a mean correction, and a variance correction:
`d_x = (d_norm − mean(d_norm) − x_hat·mean(d_norm·x_hat)) / sigma`.
`d_gamma`/`d_beta` sum over every axis except the feature axis, since both are shared across
all tokens.

### softmax_rows  — row-wise stable softmax (helper, not a Layer)
```
    - one distribution per query ROW (max/sum along the last axis only)
    - tolerates -inf entries (masked positions get exactly zero probability)
    - distinct from stable_softmax, which normalises over the whole array
```

### cross_entropy  — the loss (a function, NOT a Layer)
```
    - cross_entropy(scores, targets) -> (mean_loss, gradient_wrt_scores)
    - no parameters ; seeds backprop with gradient_wrt_scores = (softmax - onehot) / batch
```

### Adam  — the optimizer
```
    - references the layers            self.layers
    - per-param state: m, v (buffers, zeros, same shapes)
    - shared state:    t, lr, β1, β2, ε
    - step():  loop all params -> Adam update in place (reads grads fresh each step)
    - zero_grad(): reset every layer's grads
```

### CharTokenizer  — text ↔ ids (utility, NOT a Layer)
```
    - id_to_token / token_to_id : the vocabulary (incl. <pad>, <bos>, <eos>)
    - encode(text) -> [ids]   (optionally wraps with <bos>/<eos>)
    - decode([ids]) -> text   (optionally skips special tokens)
    - vocab_size
```

### Models — assembled from the components above
```
Bigram   : Embedding(vocab, d) -> Linear(d, vocab) -> cross_entropy -> Adam
           predicts next token from ONLY the current token
           trained on shift-by-one pairs: inputs = ids[:-1], targets = ids[1:]
           generate(): <bos> -> embed -> Linear -> softmax -> sample -> until <eos>

Captioner: Embedding + PatchEmbedding -> [Transformer blocks] -> Linear -> cross_entropy
           (the same skeleton; later steps slot layers into the middle)
```

**Class relationship (training-time view):**

This shows the **learning process** — `Adam` (and `cross_entropy`) exist only while training.
At **generation time** they drop out: only `CharTokenizer` + the trained model (`Bigram` →
`Embedding` + `Linear`) run, via `generate()`.

```mermaid
classDiagram
    class Layer {
        <<interface>>
        +forward(X) Y
        +backward(d_y) d_x
        +parameters() list~value_grad~
    }
    class Linear {
        +W  value
        +b  value
        +d_w grad
        +d_b grad
        +forward(X) Y
        +backward(d_y) d_x
        +parameters() list~value_grad~
    }
    class Embedding {
        +table value
        +d_table grad
        +forward(ids) rows
        +backward(d_out) None
        +parameters() list~value_grad~
    }
    class Adam {
        -layers list~Layer~
        -m buffers
        -v buffers
        -t int
        -lr, beta1, beta2, eps
        +step()
        +zero_grad()
    }
    class CharTokenizer {
        +vocab_size
        +encode(text) ids
        +decode(ids) text
    }
    class CausalSelfAttention {
        +query_projection Linear
        +key_projection Linear
        +value_projection Linear
        +attention_weights cache
        +forward(X) output
        +backward(d_out) d_x
        +parameters() list~value_grad~
    }
    class MultiHeadAttention {
        +heads list
        +output_projection Linear
        +forward(X) output
        +backward(d_out) d_x
        +parameters() list~value_grad~
    }
    class LayerNorm {
        +gamma value
        +beta value
        +d_gamma grad
        +d_beta grad
        +forward(X) Y
        +backward(d_y) d_x
        +parameters() list~value_grad~
    }
    class Bigram {
        +embedding Embedding
        +projection Linear
        +forward(ids) logits
        +backward(d_logits)
        +generate() ids
    }
    Layer <|.. Linear : implements
    Layer <|.. Embedding : implements
    Layer <|.. CausalSelfAttention : implements
    Layer <|.. MultiHeadAttention : implements
    Layer <|.. LayerNorm : implements
    CausalSelfAttention *-- Linear : Q, K, V
    MultiHeadAttention *-- CausalSelfAttention : many heads
    MultiHeadAttention *-- Linear : output projection
    Adam o--> "many" Layer : references and updates
    Bigram *-- Embedding : has
    Bigram *-- Linear : has
    Bigram ..> CharTokenizer : uses
```

**One training step (how they interact):**
```mermaid
sequenceDiagram
    participant CE as cross_entropy
    participant M as Layers (model)
    participant A as Adam
    CE->>M: d_scores (∂L/∂scores)
    M->>M: backward() fills each layer d_w, d_b
    A->>M: parameters() read value and fresh grad
    A->>A: update m and v, then bias-correct
    A->>M: value -= lr * m_hat / (sqrt(v_hat) + eps) in place
    A->>M: zero_grad() reset grads
```

**Two correctness rules:**
- **Update in place** — `value -= …` (not `value = value − …`), else the layer's `W` never changes.
- **Read grads fresh each step** — `backward()` rebinds `d_w`/`d_b` to new arrays, so `step()`
  must call `parameters()` each time; values are safe to hold, grads are not.

---

## Deliverables (the executable)

| Artifact | What it is |
|---|---|
| **`captioning.ipynb`** | the deliverable — all code + markdown, runs start→finish |
| **`model.py`** *(optional)* | from-scratch modules, imported by the notebook |
| **`weights.npy`** | trained parameters, so the demo runs without retraining |
| **`report.pdf`** | ~6-page GPT report |

Core demo cell: `caption = generate(model, image); print(caption)` + attention overlay.

---

## Build order & verification

Build bottom-up; each row reuses the ones above. Nothing with a backward pass is done until
its **gradient check passes**. (Theory for each is in [LEARNING.md](LEARNING.md).)

| Build | Verify (done when) |
|---|---|
| `numerical_gradient`, `stable_softmax` | grad of `x²` is `2x`; softmax safe for `x + 1000` |
| `Linear` (+ activation, MLP) | grad checks on `d_w/d_b/d_x` pass; MLP solves XOR |
| `cross_entropy`, `Adam` | loss falls, weights change in place, fits a learnable task; grad check passes |
| `CharTokenizer`, `Embedding`, Bigram + `generate` | tokenizer round-trips; embedding grad check; loss `< ln(vocab)`; forms letter patterns |
| Attention (causal, multi-head) ⭐ | grad checks pass; changing a future token doesn't change earlier outputs |
| `LayerNorm`, Transformer block, char-GPT | grad check; val loss drops; word-like text — ✅ **valid Track-3 project reached** |
| `patchify`, `PatchEmbedding`, joint sequence | shapes correct; image unmasked / caption causal; end-to-end grad check |
| Train captioner + `generate(image)` | clean train/val curves; captions relate to held-out images |
| Diagnostics + grounding analysis | overfit 2 examples → ~0 loss; some words ground to sensible regions |

**Safety net:** the char-GPT row already gives a submittable Track-3 text GPT; the rows below
it are the multimodal upgrade on top.
