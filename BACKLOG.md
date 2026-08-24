# Implement from scratch

## Tier 1 — Math foundations

### 1. Derivatives & partial derivatives
* **Original sources:** no origin paper — classical calculus.
* **Lectures / impl:** [3Blue1Brown — Essence of Calculus](https://www.3blue1brown.com/topics/calculus) (visual intuition) · [CS231n — backprop notes](https://cs231n.github.io/optimization-2/) (derivatives as local gradients on a graph)

### 2. Chain rule
* **Original sources:** no origin paper — classical calculus.
* **Lectures / impl:** [Karpathy — micrograd](https://github.com/karpathy/micrograd) — a tiny scalar autograd engine; the chain rule *is* the whole repo · [CS231n — backprop](https://cs231n.github.io/optimization-2/)

### 3. Matrix calculus / gradients of scalar functions
* **Original sources:** no origin paper. Best reference: [The Matrix Calculus You Need For Deep Learning](https://arxiv.org/abs/1802.01528) — Parr & Howard (2018)
* **Lectures / impl:** [explained.ai — matrix calculus](https://explained.ai/matrix-calculus/) (same text, web form)

### 4. Finite differences / gradient checking
* **Original sources:** no origin paper — standard numerical analysis.
* **Lectures / impl:** [CS231n — neural nets case study](https://cs231n.github.io/neural-networks-case-study/) (hand-rolled gradients + numeric check)

### 5. Numerical stability (float overflow, log-sum-exp)
* **Original sources:** no origin paper — numerical computing folklore.
* **Lectures / impl:** [d2l.ai](https://d2l.ai/) — softmax implementation chapter discusses the max-subtraction trick

---

## Tier 2 — The neuron and the network

### 6. Linear layer (`Y = X @ W + b`)
* **Original sources:** no single paper for the layer itself; the lineage is the perceptron / MLP.
* **Lectures / impl:** [Karpathy — micrograd](https://github.com/karpathy/micrograd) → [makemore](https://github.com/karpathy/makemore) · [CS231n case study](https://cs231n.github.io/neural-networks-case-study/)

### 7. Activations — sigmoid, tanh, ReLU, GELU
* **Original sources:** **Original paper (GELU):** [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415) — Hendrycks & Gimpel (2016). Sigmoid/tanh predate the field; ReLU has no single origin paper.
* **Lectures / impl:** [d2l.ai](https://d2l.ai/) (MLP chapter) · [nn.labml.ai](https://nn.labml.ai/) (annotated PyTorch implementations)

### 8. Backpropagation
* **Original sources:** **Original paper:** [Learning representations by back-propagating errors](https://www.cs.toronto.edu/~hinton/absps/naturebp.pdf) — Rumelhart, Hinton & Williams, *Nature* (1986)
* **Lectures / impl:** [Karpathy — micrograd](https://github.com/karpathy/micrograd) + lecture 1 of [nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) · lecture 5 there ("Becoming a Backprop Ninja") does every gradient by hand

### 9. Weight initialization (Xavier / He)
* **Original sources:** **Original paper (Xavier):** [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.html) — Glorot & Bengio (2010). **He init:** [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852) — He et al. (2015)
* **Lectures / impl:** [nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) lecture 4 (Activations & Gradients)

### 10. Vanishing / exploding gradients
* **Original sources:** **Original paper:** [On the difficulty of training Recurrent Neural Networks](https://arxiv.org/abs/1211.5063) — Pascanu et al. (2012) — also the origin of gradient clipping
* **Lectures / impl:** [nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) lecture 4

---

## Tier 3 — Output, loss, optimization

### 11. Softmax + cross-entropy (and the combined gradient)
* **Original sources:** no single origin paper; softmax-as-output traces to Bridle (1990) and information theory.
* **Lectures / impl:** [CS231n case study](https://cs231n.github.io/neural-networks-case-study/) (derives `softmax − onehot`) · [d2l.ai](https://d2l.ai/)

### 12. SGD, mini-batches, learning rate
* **Original sources:** no accessible origin paper for SGD as used here (Robbins & Monro, 1951).
* **Lectures / impl:** [d2l.ai](https://d2l.ai/) optimization chapter · [nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero)

### 13. Adam (and AdamW)
* **Original sources:** **Original paper:** [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Kingma & Ba (2014). **AdamW:** [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) — Loshchilov & Hutter (2017)
* **Lectures / impl:** [nn.labml.ai](https://nn.labml.ai/) has a line-by-line annotated Adam

### 14. Regularization — dropout, weight decay
* **Original sources:** **Original paper:** [Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/abs/1207.0580) — Hinton et al. (2012); full version: [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.jmlr.org/papers/v15/srivastava14a.html) — Srivastava et al. (2014)
* **Lectures / impl:** [d2l.ai](https://d2l.ai/) regularization chapter

### 15. Normalization — BatchNorm vs LayerNorm
* **Original sources:** **BatchNorm:** [Batch Normalization](https://arxiv.org/abs/1502.03167) — Ioffe & Szegedy (2015). **LayerNorm:** [Layer Normalization](https://arxiv.org/abs/1607.06450) — Ba, Kiros & Hinton (2016)
* **Lectures / impl:** [nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) lecture 4 implements BatchNorm by hand, backward included

---

## Tier 4 — Language modeling

### 16. Tokenization — char → BPE / WordPiece / SentencePiece
* **Original sources:** **BPE for NMT:** [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — Sennrich et al. (2015). **SentencePiece:** [SentencePiece](https://arxiv.org/abs/1808.06226) — Kudo & Richardson (2018)
* **Lectures / impl:** [nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) lecture 8 ("Let's build the GPT Tokenizer") — BPE trained from scratch

### 17. Embeddings (+ the scatter-add backward)
* **Original sources:** **Original paper (word embeddings):** [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) — Mikolov et al. (2013). Scatter-add has no paper — it is the gradient of a gather.
* **Lectures / impl:** [makemore](https://github.com/karpathy/makemore) · [nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) lecture 3

### 18. N-gram / bigram language model
* **Original sources:** no modern origin paper; traces to Shannon's information theory (1948).
* **Lectures / impl:** [nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) lecture 2 (counting bigrams, then the neural version)

### 19. Neural probabilistic language model (embedding → MLP → next token)
* **Original sources:** **Original paper:** [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/v3/bengio03a.html) — Bengio et al. (2003)
* **Lectures / impl:** [nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) lecture 3 is a direct re-implementation of this paper

### 20. Decoding — greedy, temperature, top-k, nucleus, beam search
* **Original sources:** **Nucleus (top-p):** [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) — Holtzman et al. (2019)
* **Lectures / impl:** [nanoGPT](https://github.com/karpathy/nanoGPT) `generate()` (temperature + top-k)

### 21. Evaluation — perplexity, BLEU
* **Original sources:** **BLEU:** [Bleu: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/) — Papineni et al. (2002). Perplexity has no origin paper (it is `exp(cross-entropy)`).
* **Lectures / impl:** [CS224n](https://web.stanford.edu/class/cs224n/) covers both

---

## Tier 5 — Sequence models (the road to attention)

### 22. RNN
* **Original sources:** **Original paper:** [Finding Structure in Time](https://onlinelibrary.wiley.com/doi/10.1207/s15516709cog1402_1) — Elman (1990)
* **Lectures / impl:** [d2l.ai](https://d2l.ai/) RNN chapter (from scratch, then framework) · [CS224n](https://web.stanford.edu/class/cs224n/)

### 23. Seq2Seq (encoder → fixed vector → decoder)
* **Original sources:** **Original paper:** [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — Sutskever et al. (2014)
* **Lectures / impl:** [d2l.ai](https://d2l.ai/) seq2seq chapter

### 24. LSTM
* **Original sources:** **Original paper:** [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — Hochreiter & Schmidhuber, *Neural Computation* 9(8):1735–1780 (1997)
* **Lectures / impl:** [d2l.ai](https://d2l.ai/) LSTM chapter (gate-by-gate from scratch) · [nn.labml.ai](https://nn.labml.ai/)

### 25. Additive / multiplicative attention (pre-Transformer)
* **Original sources:** **Original paper:** [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — Bahdanau et al. (2014). **Dot-product variants:** [Effective Approaches to Attention-based NMT](https://arxiv.org/abs/1508.04025) — Luong et al. (2015)
* **Lectures / impl:** [d2l.ai](https://d2l.ai/) attention chapter · [CS224n](https://web.stanford.edu/class/cs224n/)

---

## Tier 6 — Transformer

### 26. Scaled dot-product self-attention + causal mask
* **Original sources:** **Original paper:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al. (2017), §3.2
* **Lectures / impl:** [nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) lecture 7 ("Let's build GPT") · [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) · [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) for intuition

### 27. Multi-head attention
* **Original sources:** **Original paper:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762), §3.2.2
* **Lectures / impl:** [Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) (batched heads via reshape) · [nn.labml.ai](https://nn.labml.ai/)

### 28. Residual connections
* **Original sources:** **Original paper:** [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) — He et al. (2015)
* **Lectures / impl:** [d2l.ai](https://d2l.ai/) ResNet chapter

### 29. Pre-norm vs post-norm block ordering
* **Original sources:** **Original paper:** [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — Xiong et al. (2020) — why GPT uses pre-norm
* **Lectures / impl:** [nanoGPT](https://github.com/karpathy/nanoGPT) `model.py` (pre-norm block)

### 30. Positional encodings — sinusoidal, learned, RoPE
* **Original sources:** **Sinusoidal:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762), §3.5. **RoPE:** [RoFormer: Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — Su et al. (2021)
* **Lectures / impl:** [nn.labml.ai](https://nn.labml.ai/) has an annotated RoPE

### 31. The full Transformer block + decoder-only GPT
* **Original sources:** **Transformer:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762). **Decoder-only GPT:** [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — Radford et al. (2018)
* **Lectures / impl:** [minGPT](https://github.com/karpathy/minGPT) (minimal, readable) → [nanoGPT](https://github.com/karpathy/nanoGPT) (trainable) · [nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) lecture 7

### 32. Efficiency — KV caching, FlashAttention *(good to know)*
* **Original sources:** **FlashAttention:** [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Dao et al. (2022). KV caching has no origin paper — it is an engineering optimization.
* **Lectures / impl:** [nanoGPT](https://github.com/karpathy/nanoGPT) uses fused attention; [nn.labml.ai](https://nn.labml.ai/)

---

## Tier 7 — Vision & multimodal

### 33. Images as arrays, patchify, patch embedding
* **Original sources:** **Original paper:** [An Image is Worth 16x16 Words (ViT)](https://arxiv.org/abs/2010.11929) — Dosovitskiy et al. (2020)
* **Lectures / impl:** [d2l.ai](https://d2l.ai/) ViT chapter · [nn.labml.ai](https://nn.labml.ai/)

### 34. Image captioning
* **Original sources:** **Original paper:** [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555) — Vinyals et al. (2014). **With attention:** [Show, Attend and Tell](https://arxiv.org/abs/1502.03044) — Xu et al. (2015) — the ancestor of my grounding analysis
* **Lectures / impl:** [nanoVLM](https://github.com/huggingface/nanoVLM) (small vision-language model, pure PyTorch)

### 35. Contrastive image–text alignment (CLIP)
* **Original sources:** **Original paper:** [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) — Radford et al. (2021)
* **Lectures / impl:** [nn.labml.ai](https://nn.labml.ai/)

### 36. Decoder-only VLM with an image prefix *(this project's architecture)*
* **Original sources:** no single origin paper — it combines [ViT](https://arxiv.org/abs/2010.11929) patch tokens with a [decoder-only GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf); contrast with the cross-attention design of [Show, Attend and Tell](https://arxiv.org/abs/1502.03044).
* **Lectures / impl:** [nanoVLM](https://github.com/huggingface/nanoVLM) — closest reference implementation

---

## Cross-cutting resources

* [Karpathy — Neural Networks: Zero to Hero](https://github.com/karpathy/nn-zero-to-hero) — 8 lectures, micrograd → GPT → tokenizer. The single closest match to this backlog.
* [d2l.ai — Dive into Deep Learning](https://d2l.ai/) — interactive book; nearly every item above has a from-scratch chapter.
* [nn.labml.ai](https://nn.labml.ai/) — line-by-line annotated PyTorch implementations.
* [CS231n](https://cs231n.github.io/) — backprop and gradient mechanics done by hand.
* [CS224n](https://web.stanford.edu/class/cs224n/) — NLP from word vectors to LLMs.
* [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — the paper as runnable code.
