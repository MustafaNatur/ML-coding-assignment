# Learning Plan — Theory Companion

The **study guide** behind [IMPLEMENTATION.md](IMPLEMENTATION.md) (same step numbers). Each
step has:

- **Must-learn** — the theory you need to implement that step
- **Good to know (career)** — extra depth that pays off in interviews / real ML work
- **Practice** — general exercises to *feel* the concept, beyond our specific project

You said you want to over-read — the "Good to know" and "Practice" columns are where to
spend that extra energy. Skipping them still lets you finish the project.

---

## Step 0 — Foundations & the Gradient Checker

**Must-learn**
- Chain rule, partial derivatives, gradients of scalar functions
- Finite differences (central: `(f(x+h)−f(x−h))/2h`) and why they approximate gradients
- Numerical stability: floating point, overflow, the **log-sum-exp** trick

**Good to know (career)**
- Jacobians and the Jacobian-vector-product view (what autograd frameworks actually do)
- Forward-mode vs. reverse-mode autodiff (why deep learning uses reverse mode)
- Condition number / catastrophic cancellation (why naive softmax overflows)

**Practice**
- Hand-derive gradients of `x²`, `sin(x)`, `1/x`, `log(x)` and check them numerically
- Implement a tiny scalar autograd (à la micrograd) with `+`, `*`, `tanh`
- Write a stable softmax and prove it equals the naive one on small inputs

## Step 1 — Neural Nets & Backprop

**Must-learn**
- Linear layer `Y = XW + b`; matrix shapes and how they line up
- Backprop through a layer: gradients for `W`, `b`, `X`
- Activations: sigmoid, tanh, ReLU, **GELU**; their derivatives
- Weight initialization (why scale matters: Xavier/He)

**Good to know (career)**
- Vanishing/exploding gradients; how ReLU, init, and normalization mitigate them
- Universal approximation theorem (what one hidden layer can/can't do)
- Dead ReLUs, activation saturation

**Practice**
- Implement a Linear + ReLU MLP; classify a 2-D toy dataset (moons/circles)
- Solve XOR (proof a hidden layer is needed)
- Break init on purpose (all zeros, huge values) and watch training fail

## Step 2 — Output, Loss & Optimization

**Must-learn**
- Softmax as a probability distribution
- Cross-entropy loss; the clean combined gradient `softmax − onehot` (derive it)
- Gradient descent: SGD, learning rate, mini-batches
- **Adam**: momentum + adaptive lr, bias correction

**Good to know (career)**
- Optimizers landscape: SGD+momentum, RMSProp, Adam, AdamW (decoupled weight decay)
- Learning-rate schedules: warmup, cosine/linear decay
- Regularization: L2, dropout, early stopping, label smoothing
- Metrics: accuracy vs. loss vs. **perplexity** (for language models)

**Practice**
- Train a softmax classifier on sklearn `digits`; plot loss/accuracy
- Compare SGD vs. Adam convergence on the same task
- Sweep learning rate (too small / good / too big) and observe the curves

## Step 3 — Language Modeling Basics

**Must-learn**
- Next-token prediction: `p(w_t | w_1..w_{t-1})`
- Tokenization: char-level, vocab, special tokens (`<bos>`/`<eos>`/`<pad>`)
- Embeddings: token id → learned vector; embedding backward (scatter-add)
- Teacher forcing (train) vs. autoregressive generation (inference)
- Positional embeddings (why order matters)

**Good to know (career)**
- Sub-word tokenization: BPE, WordPiece, SentencePiece (what real LLMs use)
- Word2Vec / embedding geometry (king − man + woman ≈ queen)
- N-gram models and their limits (motivates neural LMs)
- Sampling: greedy, temperature, top-k, top-p (nucleus), beam search

**Practice**
- Build a char tokenizer; encode/decode round-trip
- Train a bigram/MLP next-char model on a text file; generate samples
- Experiment with temperature and watch diversity vs. coherence

## Step 4 — Attention ⭐ *(deepest — the core skill)*

**Must-learn**
- Query / Key / Value: what each represents
- Scaled dot-product attention `softmax(QKᵀ/√d)V`; why divide by √d
- Causal masking (−∞ before softmax) → no peeking at the future
- Multi-head attention: parallel heads, concat, output projection
- **Backprop through attention**: through V-matmul, softmax, QKᵀ (the hardest backward)

**Good to know (career)**
- Self- vs. cross-attention (encoder-decoder); we use self-attention over a joint sequence
- Attention as soft dictionary lookup / kernel smoothing
- O(n²) cost → why context windows are limited; efficient attention (FlashAttention, etc.)
- Positional variants: sinusoidal vs. learned vs. **RoPE**
- KV-caching for fast generation

**Practice**
- Implement single-head attention; visualize an attention matrix as a heatmap
- Add a causal mask; verify no future leakage empirically
- Extend to multi-head; gradient-check the whole thing

## Step 5 — Normalization, Blocks & Deep Nets

**Must-learn**
- LayerNorm: normalize across features; learnable scale/shift; its backward pass
- Residual connections `x + sublayer(x)`; why they help gradient flow
- Pre-norm vs. post-norm block ordering (GPT = pre-norm)
- The feed-forward/MLP sub-layer (×4 expand → GELU → project)
- Stacking N blocks for depth

**Good to know (career)**
- BatchNorm vs. LayerNorm vs. RMSNorm (when each is used)
- Why Transformers replaced RNNs/LSTMs (parallelism, long-range deps)
- Residual networks (ResNet) and the degradation problem
- Parameter counting (per block, per model) — sizing intuition

**Practice**
- Implement LayerNorm; gradient-check the tricky backward
- Assemble one Transformer block; gradient-check end to end
- Build a small char-GPT on Tiny Shakespeare; generate text

## Step 6 — Vision & Multimodal

**Must-learn**
- Images as `(H,W,C)` arrays; normalization
- Patches-as-tokens (the ViT idea); patch embedding = one Linear
- Mixing image + text tokens in one sequence

**Good to know (career)**
- CNNs vs. ViT (inductive biases, data efficiency) — even though we use patches
- Multimodal models: CLIP (contrastive), captioning lineage (Show and Tell → Show, Attend
  and Tell), modern VLMs
- Grounding: connecting words to visual regions (this is Prof. Bruni's field)
- Cross-modal fusion strategies (early/late/prefix conditioning)

**Practice**
- Patchify an image and reconstruct it from patches (sanity check)
- Visualize patch embeddings; feed image tokens into your char-GPT

## Step 7 — Training a Real System

**Must-learn**
- Mini-batching sequences; padding/masking variable lengths
- Loss masking (compute loss only where it's valid)
- Decoding strategies at generation time

**Good to know (career)**
- Data pipelines, train/val/test hygiene, avoiding leakage
- Overfitting vs. underfitting diagnosis; capacity vs. data
- Compute/memory trade-offs; why we downscale and subset
- Evaluation of generation: BLEU/CIDEr (concept), and their limits

**Practice**
- Train your captioner on a small subset; plot train vs. val loss
- Generate captions with greedy vs. temperature; compare
- Deliberately overfit a tiny set to confirm the pipeline works

## Step 8 — Diagnostics, Analysis & Interpretability

**Must-learn**
- Gradient checking a full model
- Reading learning curves (over/underfitting signatures)
- Perplexity as an LM metric
- Attention visualization / overlay

**Good to know (career)**
- Ablation methodology (change one variable, measure)
- Interpretability basics: attention maps, saliency, probing
- Honest reporting of failure cases (a research skill)
- Reproducibility (seeds, config logging)

**Practice**
- Overlay text→image attention on a picture; inspect grounding
- Run a small ablation (heads/layers/lr) and tabulate
- Write a paragraph interpreting what the model learned and where it fails

---

## Two pillars *(if time is short)*
1. **Backprop + chain rule** (Steps 0–1) — can't write any backward pass without it
2. **Attention + its backprop** (Step 4) — the heart and the hardest part

## Key resources
- Karpathy — *Zero to Hero* (micrograd → makemore → *Let's build GPT*) — closest to our project
- Parr & Howard — *The Matrix Calculus You Need for Deep Learning*
- 3Blue1Brown — *Essence of Linear Algebra* + *Neural Networks* series
- Jay Alammar — *The Illustrated Transformer*
- Vaswani et al. — *Attention Is All You Need* (§3)
- Dosovitskiy et al. — *An Image is Worth 16×16 Words* (ViT)
- Xu et al. — *Show, Attend and Tell* (captioning + attention grounding)
