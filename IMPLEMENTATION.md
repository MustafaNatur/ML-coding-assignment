# Implementation — character-level GPT from scratch in NumPy

The build spec. Theory for each component is in [LEARNING.md](LEARNING.md); components to
re-implement solo later are tracked in [BACKLOG.md](BACKLOG.md).

> **Golden rule:** nothing with a backward pass is done until a **finite-difference gradient
> check passes**.

---

## Task

Build a **decoder-only Transformer (GPT) that generates text one character at a time**, entirely
from scratch in NumPy — Track 3 of the assignment.

- **Dataset:** Tiny Shakespeare, ~1.1 M characters of play text, 90/10 train/validation split.
- **Model:** 619,969 parameters — `d_model=128`, 4 heads, 3 blocks, 64-character context.
- **Deliverable:** a Jupyter notebook that runs start→finish + a ~6-page report.
- **Constraint:** the network is from scratch; only NumPy for the model. Matplotlib for plots,
  tokenizer self-written, Python stdlib for data loading.

**Scope note.** This branch is deliberately limited to the text GPT, which is a complete Track-3
project on its own. The multimodal image-captioning extension (patch embeddings, split mask,
Flickr8k, grounding analysis) lives on `main` and is not part of this deliverable.

---

## Data flow

Where the *data* goes, end to end — the loop **around** the model. The architecture diagram in the
next section zooms into the single `CharGPT.forward` box below. Names match the notebook.

### Training

```mermaid
flowchart TD
    subgraph PREP["prepare once"]
        FILE["tinyshakespeare.txt<br/>~1.1M characters"] --> ENCODE["CharTokenizer.encode"]
        ENCODE --> IDS["all_token_ids<br/>(N,) int"]
        IDS --> SPLIT["split at 90%"]
        SPLIT --> TRAIN["train_token_ids<br/>first 90%"]
        SPLIT --> VAL["validation_token_ids<br/>last 10%"]
    end

    subgraph STEP["one training step -- repeated 16,000x"]
        WINDOW["random window<br/>(block_size+1,)"]
        INPUTS["inputs: window minus its last id<br/>(T,)"]
        TARGETS["targets: the same window,<br/>shifted one to the left (T,)"]
        FORWARD["CharGPT.forward"]
        LOGITS["logits<br/>(T, vocab)"]
        LOSSFN["cross_entropy"]
        DLOGITS["d_logits<br/>(T, vocab)"]
        BACKWARD["CharGPT.backward"]
        GRADS["gradient buffers<br/>d_w, d_b, d_table, d_gamma, d_beta"]
        UPDATE["Adam.step then zero_grad"]
    end

    PARAMS[("parameters<br/>updated in place")]

    TRAIN --> WINDOW
    WINDOW --> INPUTS
    WINDOW --> TARGETS
    INPUTS --> FORWARD --> LOGITS --> LOSSFN
    TARGETS --> LOSSFN
    LOSSFN --> DLOGITS --> BACKWARD --> GRADS --> UPDATE --> PARAMS
    PARAMS -. "read by the next step" .-> FORWARD

    LOSSFN --> TRACE["train_loss_history"]
    VAL --> EVAL["estimate_loss<br/>20 windows, forward only"]
    PARAMS -. "every 1000 steps" .-> EVAL
    EVAL --> VALHIST["validation_loss_history"]
    TRACE --> CURVES["loss curves, perplexity"]
    VALHIST --> CURVES

    classDef data fill:#e3f2fd,stroke:#1565c0;
    classDef state fill:#fff3e0,stroke:#e65100;
    class FILE,IDS,TRAIN,VAL,WINDOW,INPUTS,TARGETS data;
    class GRADS,PARAMS state;
```

**What this makes explicit**
- **Targets enter in exactly one place** — `cross_entropy`. The model itself never sees them, which
  is only legitimate because the mask is causal.
- **The only state passed from backward to the update** is the per-layer gradient buffer. `backward`
  fills it, `Adam.step()` reads it, `zero_grad()` clears it; nothing else survives a step except the
  parameters themselves.
- **The validation branch is forward-only** — it shares the window/loss machinery but never reaches
  `backward`, so held-out data can never touch a parameter.
- **One window per step**: there is no batch axis anywhere in the diagram. That is the single
  biggest performance limitation of this implementation (see *Known limitation* under Results).

### Generation

After training, only the tokenizer and the model run. Each pass produces logits for *every*
position, but the loop keeps only the last row — the prediction for the character that comes next.

```mermaid
flowchart TD
    PROMPT["prompt text<br/>(empty by default)"] --> GENC["CharTokenizer.encode"]
    GENC --> CTX["generated_ids:<br/>bos + prompt ids"]
    CTX --> CROP["crop to the last<br/>max_sequence_length ids"]
    CROP --> GFWD["CharGPT.forward"]
    GPARAMS[("trained parameters")] -. "frozen" .-> GFWD
    GFWD --> GLOGITS["logits<br/>(T, vocab)"]
    GLOGITS --> LAST["last row / temperature<br/>(vocab,)"]
    LAST --> PROBS["softmax over the vocabulary"]
    PROBS --> SAMPLE["sample one id"]
    SAMPLE --> APPEND["append to generated_ids"]
    APPEND -. "until max_new_tokens or eos" .-> CROP
    APPEND --> DECODE["CharTokenizer.decode"]
    DECODE --> TEXT["generated text"]

    classDef data fill:#e3f2fd,stroke:#1565c0;
    classDef out fill:#fff3e0,stroke:#e65100;
    class PROMPT,CTX,CROP data;
    class TEXT,GPARAMS out;
```

- **The crop is not an optimization, it is a hard limit**: the position embedding table has exactly
  `max_sequence_length` rows, so a longer context has no position vector to look up. Everything
  older than 64 characters is simply forgotten.
- **No key/value cache**: each new character re-runs the full forward pass over the whole context
  instead of reusing the previous one, so a 400-character sample costs 400 forward passes. This is
  the reason generation is slower than it looks for a model this small.

---

## Architecture overview

The forward data-flow. It is shared by both modes and differs only at the very end: **training**
feeds predictions into cross-entropy; **generation** samples the next character and loops.

Shapes use `d_model=128`, context `T ≤ 64`, vocabulary 65.

```mermaid
flowchart TD
    IDS["character ids (T,)"] --> TEMB["Token Embedding<br/>(vocab, 128)"]
    POSIDS["positions 0..T-1"] --> PEMB["Position Embedding<br/>(64, 128)"]

    TEMB -->|"(T, 128)"| SUM["+ (sum)"]
    PEMB -->|"(T, 128)"| SUM

    SUM -->|"(T, 128)"| BLK["N x TransformerBlock<br/>LayerNorm, MultiHeadAttention, +residual<br/>LayerNorm, FeedForward, +residual"]
    BLK -->|"(T, 128)"| FLN["Final LayerNorm"]
    FLN --> OUT["Output Linear (LM head)"]
    OUT -->|"(T, vocab)"| SM["Softmax"]

    SM --> CE["Cross-Entropy<br/>(training)"]
    SM --> GEN["Sample next character<br/>(generation)"]

    classDef inp fill:#e3f2fd,stroke:#1565c0;
    classDef out fill:#fff3e0,stroke:#e65100;
    class IDS,TEMB,POSIDS,PEMB inp;
    class CE,GEN out;
```

## Components & connections

| Component | Responsibility | In → Out | Connects |
|---|---|---|---|
| **CharTokenizer** | text ↔ integer ids; owns the vocabulary (the corpus's characters) | text → `(T,)` | → Token Embedding |
| **Token Embedding** | lookup table: id → learned vector | `(T,)` → `(T,128)` | → sum |
| **Position Embedding** | lookup table: *position* → learned vector; supplies word order | `(T,)` → `(T,128)` | → sum |
| **Sum** | "what" + "where" in one representation per position | → `(T,128)` | → blocks |
| **LayerNorm** | normalize each token across features; stabilizes depth | `(T,128)` → `(T,128)` | ×2 per block + final |
| **MultiHeadAttention** | mix information **across** positions, causally | `(T,128)` → `(T,128)` | sub-layer 1 of each block |
| **FeedForward** | transform **each** token independently (×4 width) | `(T,128)` → `(T,128)` | sub-layer 2 of each block |
| **ResidualSublayer** | `X + branch(norm(X))`; keeps gradients flowing | same shape | wraps both sub-layers |
| **TransformerBlock ×N** | one round of "attend, then think" | `(T,128)` → `(T,128)` | chained N times |
| **Final LayerNorm** | normalize before the output projection | `(T,128)` → `(T,128)` | → LM head |
| **Output Linear** (LM head) | project each token to vocabulary logits | `(T,128)` → `(T,vocab)` | → Softmax |
| **cross_entropy** *(train)* | error vs. the true next character | → scalar loss | seeds backprop |
| **generate** *(inference)* | sample, append, repeat until the token cap | → text | the demo |

**Key connections**
- **Token + position are summed**, so backward sends the *same* gradient to both tables.
- **The positional table's size *is* the context window**, so `generate()` crops its input to the
  last `max_sequence_length` characters.
- **The mask is causal**: position `i` attends only to `≤ i`. That is what makes training on all
  `T` positions at once legitimate.
- **One block = attend, then process** — attention shares information *between* tokens, the MLP
  transforms *each* token; residual + LayerNorm keep the stack trainable.

## Input / Output

- **Training input:** one window of `block_size + 1` ids → `inputs = window[:-1]`,
  `targets = window[1:]` (teacher forcing).
- **Training output:** scalar cross-entropy → gradients into every layer.
- **Inference input:** a prompt string (defaults to a newline).
- **Inference output:** generated text, one character at a time, up to `max_new_tokens`.

**Runtime (generation) flow**
```
start: prompt ids
loop max_new_tokens times:
   crop to the last max_sequence_length ids
   forward pass -> take probabilities at the LAST position
   sample (temperature) -> append
```

| Iter | Model input | Predicts |
|---|---|---|
| 1 | `R O M E O :` | `\n` |
| 2 | `R O M E O : \n` | `W` |
| 3 | `… W` | `h` |

---

## Software design (components & responsibilities)

Four kinds of component: **layers compute**, the **loss** scores and seeds backprop, the
**optimizer** updates, the **tokenizer** converts text ↔ ids. Every layer shares one base class,
so the optimizer treats them all identically.

### Layer — the base class
```
class Layer
    - parameter_names : tuple[str, ...]   # names of arrays this layer OWNS; grad of `W` is `d_W`
    - children()      -> [Layer, ...]     # the layers this one is COMPOSED of
    - parameters()    -> [(value, grad), ...]   own parameters, then children's, recursively
    - zero_grad()     -> clears the whole subtree
    - forward(), backward()               # supplied by each subclass
```
**No subclass implements `parameters()` or `zero_grad()`.** A layer declares either the arrays it
owns (`Linear`, `Embedding`, `LayerNorm`) or the layers it is built from (everything else), and the
base class derives the rest. This is the code a hand-written model gets wrong *silently* — forget a
sub-layer in `parameters()` and it simply never trains — so it is written once.

### Sequential — a chain of layers
```
class Sequential(Layer)
    - __init__(*layers) ; children() -> those layers ; [i] -> layer i
    - forward(X)    -> run the layers in order
    - backward(d_y) -> run them in REVERSE, threading the gradient
```
Used for `FeedForward`, `TransformerBlock`, the block stack, and the output head — none of which
then needs a forward or backward pass of its own.

### ResidualSublayer — the pre-norm wrapper
```
class ResidualSublayer(Layer)
    - __init__(norm, branch) ; children() -> [norm, branch]
    - forward(X)    -> X + branch(norm(X))
    - backward(d_y) -> d_y + norm.backward(branch.backward(d_y))
```
The pattern appears twice per block, so it is written once. The bare `d_y` term *is* the skip path:
the gradient arriving at an addition flows to both branches unchanged, which is what keeps a deep
stack trainable.

### Linear — `Y = X @ W + b`
```
    - params: W (n_in, n_out), b (n_out)   ; grads: d_W, d_b
    - init  : W ~ N(0,1) / sqrt(n_in)  (Xavier-style), b = 0
    - forward(X)    -> X @ W + b           ; caches X
    - backward(d_y) -> d_x                 ; d_W = X.T @ d_y, d_b = sum(d_y)
```
**The `1/sqrt(n_in)` scaling is load-bearing.** Each output sums `n_in` products, so unscaled
standard-normal weights grow activations by ~`sqrt(n_in)` per layer. With plain `standard_normal`
the GPT started at loss ≈14.8 instead of `ln(vocab)` ≈3.4 and stalled around 2.5 emitting
gibberish; with the scaling it trains normally. **Every gradient check still passed while it was
broken** — the gradients were right, the *scale* was wrong.

### Embedding — id → learned vector
```
    - param: table (vocab_size, d)   the ONE parameter (no bias) ; grad: d_table
    - init : table ~ N(0, 0.02)      small, as GPT does
    - forward(ids)    -> table[ids]  row lookup ; caches ids
    - backward(d_out) -> None        scatter-add into d_table (np.add.at)
```
Two differences from Linear: backward is a **scatter-add** (a repeated id accumulates gradient
from every occurrence — plain assignment would overwrite), and it returns **no input gradient**,
since integer ids are not differentiable and it is always the first layer. Used twice: once for
tokens, once for positions.

### CausalSelfAttention — mixing information across positions
```
    - params: three Linear projections (query, key, value) -> Q, K, V
    - forward(X)      -> softmax(Q @ K.T / sqrt(d_head) + causal_mask) @ V
                         X (T, d_model) -> (T, d_head) ; caches Q, K, V, weights
    - backward(d_out) -> d_x  (T, d_model)
```
The hardest chain in the project — four links: the weighted sum `weights @ V`, the row-wise
**softmax Jacobian** `∂L/∂s = p * (∂L/∂p − Σ(∂L/∂p·p))`, the `1/√d_head` scaling, then `Q Kᵀ`.
Because X feeds all three projections, `d_x` is the **sum** of the three paths. Masked entries
have `p = 0`, so no gradient leaks to the future.

### MultiHeadAttention — several attention patterns at once
```
    - h heads (each d_head = d_model / h) + output_projection Linear(d_model, d_model)
    - forward(X)      -> concat(head outputs) -> output_projection
    - backward(d_out) -> split the gradient per head, SUM their d_x
```
Rejects a `d_model` not divisible by `number_of_heads` rather than misbehaving quietly.

### LayerNorm — keeping activations at a stable scale
```
    - params: gamma (d_model,) scale [init 1], beta (d_model,) shift [init 0]
    - forward(X)    -> gamma * (X - mean) / sqrt(var + eps) + beta
                       mean/var over the LAST axis (one token), not the batch
    - backward(d_y) -> d_x
```
Second-hardest derivation: `mu` and `sigma` depend on **every** feature of a token, so one
feature moves all of that token's outputs. Hence three terms —
`d_x = (d_norm − mean(d_norm) − x_hat·mean(d_norm·x_hat)) / sigma`. `d_gamma`/`d_beta` sum over
every axis except the feature axis, since both are shared across tokens.

### Gelu — smooth activation (no parameters)
```
    - gelu(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 x^3) ))
    - forward/backward only; FeedForward owns the parameters around it
```

### FeedForward — the MLP sub-layer
```
class FeedForward(Sequential)
    - Linear(d_model -> 4*d_model) -> Gelu -> Linear(4*d_model -> d_model)
```
Attention mixes information **between** tokens; this transforms **each token independently**.
The ×4 expansion is where most of the parameters live. A pure chain, so `Sequential` supplies
everything but the constructor.

### TransformerBlock — one pre-norm decoder block
```
class TransformerBlock(Sequential)
    - ResidualSublayer(LayerNorm, MultiHeadAttention)   # h = X + attention(norm(X))
    - ResidualSublayer(LayerNorm, FeedForward)          # Y = h + feed_forward(norm(h))
    - .attention -> the MultiHeadAttention, for inspecting a trained block
```
Pre-norm (normalizing *inside* the residual branch) keeps the residual stream un-normalized end to
end, which trains far more stably at depth than post-norm.

### CharGPT — the assembled model
```
    - token_embedding + position_embedding -> blocks (Sequential)
                                           -> output_head (Sequential: LayerNorm, Linear)
    - forward(token_ids (T,))    -> logits (T, vocab_size)
    - backward(d_logits)         -> None (gradients land in the layers)
    - parameter_count()          -> trainable scalars
    - generate(tokenizer, ...)   -> text; CROPS context to max_sequence_length
```
The embedding **sum** is the only step that is not a chain, so it is the only forward code the
model writes; everything after it is `output_head(blocks(hidden))`, and backward is the same line
read upwards.

### cross_entropy — the loss (a function, NOT a Layer)
```
    - cross_entropy(scores, targets) -> (mean_loss, gradient_wrt_scores)
    - no parameters ; gradient_wrt_scores = (softmax - onehot) / batch
```

### Adam — the optimizer
```
    - Adam(model) ; per-param state m, v ; shared state t, lr, β1, β2, ε
    - step():      loop model.parameters() -> update in place, reading grads FRESH each step
    - zero_grad(): model.zero_grad()
```
Takes **one** layer, not a list: `Layer.parameters()` already flattens the whole tree, so the
optimizer never needs to know the model's shape.

### CharTokenizer — text ↔ ids (utility, NOT a Layer)
```
    - id_to_token / token_to_id : the sorted set of characters in the corpus (65)
    - encode(text) -> [ids] ; decode([ids]) -> text ; vocab_size
```
**No `<bos>`/`<eos>`/`<pad>`.** The corpus is one continuous stream sampled by random windows, so
there is no document boundary to mark, and with a single sequence per step there is nothing to pad.
Special tokens would also never appear in training, leaving their embedding rows untrained —
`generate()` therefore starts from a real prompt instead.

### Helpers
```
    numeric_gradient(f, x)              central finite differences -- the verifier for every backward
    relative_gradient_error(...)        swap in a perturbed parameter array, differentiate the loss
                                        numerically, compare with the stored analytic gradient
    softmax_rows(scores)                ROW-wise softmax; tolerates -inf (masked) entries.
                                        Used by attention and by generate() on the last logit row.
    sample_window / window_loss         one random (inputs, targets) pair and its loss --
                                        the shared unit of training AND validation
    build_model(d_model, heads, blocks)  the model under study; the sweep varies its arguments
```

**Class relationship (training-time view).** `Adam` and `cross_entropy` exist only while
training; at generation time only `CharTokenizer` + the trained `CharGPT` run.

```mermaid
classDiagram
    class Layer {
        <<base class>>
        +parameter_names
        +children() list~Layer~
        +parameters() list~value_grad~
        +zero_grad()
    }
    class Sequential {
        +layers list~Layer~
        +forward(X) Y
        +backward(d_y) d_x
    }
    class ResidualSublayer {
        +norm LayerNorm
        +branch Layer
        +forward(X) X_plus_branch
    }
    class Linear {
        +W value
        +b value
        +d_W grad
        +d_b grad
    }
    class Embedding {
        +table value
        +d_table grad
        +backward(d_out) None
    }
    class LayerNorm {
        +gamma value
        +beta value
    }
    class Gelu {
        <<no parameters>>
    }
    class CausalSelfAttention {
        +query_projection Linear
        +key_projection Linear
        +value_projection Linear
        +attention_weights cache
    }
    class MultiHeadAttention {
        +heads list
        +output_projection Linear
    }
    class FeedForward {
        <<Sequential>>
        Linear, Gelu, Linear
    }
    class TransformerBlock {
        <<Sequential>>
        two ResidualSublayers
        +attention MultiHeadAttention
    }
    class CharGPT {
        +token_embedding Embedding
        +position_embedding Embedding
        +blocks Sequential
        +output_head Sequential
        +generate() text
        +parameter_count() int
    }
    class CharTokenizer {
        +vocab_size
        +encode(text) ids
        +decode(ids) text
    }
    class Adam {
        -model Layer
        -m buffers
        -v buffers
        +step()
        +zero_grad()
    }
    Layer <|-- Sequential : extends
    Layer <|-- ResidualSublayer : extends
    Layer <|-- Linear : extends
    Layer <|-- Embedding : extends
    Layer <|-- LayerNorm : extends
    Layer <|-- Gelu : extends
    Layer <|-- CausalSelfAttention : extends
    Layer <|-- MultiHeadAttention : extends
    Layer <|-- CharGPT : extends
    Sequential <|-- FeedForward : extends
    Sequential <|-- TransformerBlock : extends
    CausalSelfAttention *-- Linear : Q, K, V
    MultiHeadAttention *-- CausalSelfAttention : many heads
    ResidualSublayer *-- LayerNorm : norm
    TransformerBlock *-- ResidualSublayer : attention, then feed-forward
    CharGPT *-- Embedding : token + position
    CharGPT *-- TransformerBlock : N blocks
    CharGPT ..> CharTokenizer : uses
    Adam o--> Layer : walks parameters() and updates
```

**One training step**
```mermaid
sequenceDiagram
    participant CE as cross_entropy
    participant M as CharGPT layers
    participant A as Adam
    CE->>M: d_logits (∂L/∂logits)
    M->>M: backward() fills every layer's grads
    A->>M: parameters() read value and fresh grad
    A->>A: update m and v, then bias-correct
    A->>M: value -= lr * m_hat / (sqrt(v_hat) + eps) in place
    A->>M: zero_grad() reset grads
```

**Two correctness rules**
- **Update in place** — `value -= …`, never `value = value − …`, or the layer's `W` never changes.
- **Read grads fresh each step** — `backward()` rebinds grads to new arrays, so `step()` must
  call `parameters()` every time; values are safe to hold, grads are not.

---

## Build order & verification

Built bottom-up; each row reuses the ones above.

| ✓ | Component | Verified by |
|---|---|---|
| ✅ | `numeric_gradient`, `softmax_rows` | grad of `x²` is `2x`; softmax safe for `x + 1000` and for `-inf` |
| ✅ | `Linear` | grad checks on `d_W / d_b / d_x` |
| ✅ | `cross_entropy`, `Adam` | loss falls, weights change in place, fits a learnable task |
| ✅ | `CharTokenizer` | round-trips; one id per character |
| ✅ | `Embedding` | grad check; **scatter-add accumulates** on a repeated id; returns `None` |
| ✅ | `CausalSelfAttention` ⭐ | grad checks on `W_q/W_k/W_v/X`; upper triangle exactly 0; a future token cannot change earlier outputs |
| ✅ | `MultiHeadAttention` | grad checks incl. the **last** head (proves the per-head split lines up); stays causal; rejects indivisible `d_model` |
| ✅ | `LayerNorm` | grad checks incl. the three-term `d_x`; rows zero-mean/unit-variance; invariant to per-token shift and scale |
| ✅ | `Gelu`, `FeedForward` | `∂Gelu/∂X` matches numeric; ×4 hidden width; per-token (row 0 unaffected by row 1) |
| ✅ | `TransformerBlock` | grad check through both residuals; stays causal; **zeroed branches → identity** (residual proven) |
| ✅ | `CharGPT` | grad check incl. **both** embedding tables; rejects over-long sequences; same token differs by position |
| ✅ | Training on Tiny Shakespeare | starts near `ln(vocab)`; val loss well below baseline; readable words |

---

## Results

All figures below are produced by the notebook, which runs start→finish in ~3 minutes with 0 errors.
The assembled model is gradient-checked end to end in the Demonstration cell: relative error around
`1e-07` through the character embedding table, below `1e-09` through an attention query weight, and
`0.00e+00` leakage from future tokens.

**CharGPT on Tiny Shakespeare** — 10 % held-out validation tail, uniform baseline
`ln(65)` = 4.174 nats/char (perplexity 65):

| Metric | Value |
|---|---|
| Parameters | 619,969 |
| Train loss | 3.434 → 1.718 |
| **Validation loss** | 2.493 → **1.821** nats/char |
| **Validation perplexity** | **6.18** — 10.5× better than random guessing |
| Train/val gap | 0.127 nats → generalizing, not memorizing |

~6.2 perplexity means the model is effectively choosing among ~6 characters instead of 65.
Qualitatively the samples carry real English words, play formatting (speaker lines, line breaks)
and plausible punctuation, while sentences stay semantically loose — the expected outcome at this
size, which the assignment explicitly anticipates for Track 3.

**Hyperparameter finding.** The learning rate dominates, for a reason specific to this
implementation: with **one window per step** the gradient estimate is noisy, so `lr=5e-3` was the
worst setting tested (2.660 after 2000 steps, +0.30 over the baseline's 2.365) while `lr=5e-4`
reached 1.82 over the full run. Multiple heads beat a single head at equal parameter count
(2.365 vs 2.484), and halving `d_model` loses ground (2.458).

Results move slightly between runs despite fixed seeds — floating-point matrix products are not
associative, and over thousands of steps that is enough to change generated text entirely. Read
these numbers to two decimals and treat the single-seed sweep as a filter, not a ranking.

⚠️ **The 2000-step sweep ranks early learning speed, not final quality.** The 1-block model looks
best there (2.336 vs the 3-block baseline's 2.365) purely because smaller models converge faster
early; over the full budget the 3-block model reaches 1.82. A short sweep is a cheap way to rule
out clearly bad settings, not a way to pick the best architecture.

**Known limitation — no batching.** Every layer takes a **single sequence** `(T, d_model)`, which
kept the attention and LayerNorm backward passes directly gradient-checkable. The cost is
sample efficiency: one window per step instead of a batch. Adding a leading batch axis is the
obvious next optimization; it is not required for this deliverable.

---

## Deliverables

| Artifact | What it is |
|---|---|
| **`notebooks/Implementation.ipynb`** | the deliverable — all code + markdown, runs start→finish |
| **`notebooks/Theory.ipynb`** | the theory write-ups behind each component |
| **`notebooks/data/tinyshakespeare.txt`** | the corpus (auto-downloaded if absent) |
| **`report.pdf`** | ~6-page GPT report *(to write)* |

Core demo:
```python
model.generate(tokenizer, max_new_tokens=400, temperature=0.8, prompt="ROMEO:")
```
