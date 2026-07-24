# Image Captioning with a Transformer Built from Scratch

**Track 3 (GPT Report)** · Foundations of Machine Learning · Group Project

---

## One-line idea

A decoder-only Transformer (nanoGPT-style) that looks at an image and generates a
caption word-by-word — implemented **from scratch in NumPy**. It extends next-token
language modeling by conditioning on an image, bridging **Computer Vision** and **NLP**.

## Why this topic

I'm genuinely interested in **both CV and NLP**, and this project is the natural point
where they meet: teaching a model to *ground* language in what it sees. It keeps the full
Track-3 architecture (attention, backprop, training loop from scratch) while adding one
meaningful multimodal twist — so it's ambitious but built on the course's core material,
not around it.

## Fit with course topics

- Feedforward networks, backpropagation, gradient descent — **all from scratch in NumPy**
- Decoder-only Transformer: attention, LayerNorm, embeddings, softmax, Adam
- Next-token language modeling (captioning = next-token prediction conditioned on an image)
- Hyperparameter search, learning-curve diagnostics, honest analysis of a small model

## Tech stack

- **NumPy** — all model math, forward + backward passes, optimizer
- **Matplotlib / Seaborn** — loss curves, attention visualizations
- **scikit-learn** — preprocessing / splits only
- **Tokenizer** — character-level (self-written) → word-level as a stretch
- **Dataset: Flickr8k** — 8k real photos, 40k human captions; images downscaled to
  32–64px, ~2k-image subset for CPU training

## Goals & how they are tested

| Goal | How it is tested |
|---|---|
| Correct implementation | Finite-difference **gradient check** on every module |
| Learns to caption | Training/validation **cross-entropy loss curves** trend down |
| Generates real captions | **Qualitative** captions on held-out images |
| Grounding analysis | Overlay **attention weights on the image** — does "dog"/"red"/"left" attend to the right region? |

## Architecture (overview)

```
IMAGE (64x64x3)
   -> split into patches -> linear patch-embedding -> 64 image tokens (d_model)
CAPTION TOKENS -> token embedding + positional embedding
   -> concatenate: [image tokens | <bos> w1 w2 ... wn]
   -> N x Transformer blocks (causal self-attention + MLP + LayerNorm)
   -> final LayerNorm -> linear -> softmax over vocab
   -> cross-entropy on caption tokens only
```

Starter dimensions (deliberately tiny, CPU-trainable): `d_model=128`, `heads=4`,
`layers=3–4`, context = 64 image + ~40 text tokens, **~1–3M parameters**.

## Scope control

Deliberately tiny model (~1–3M params, 3–4 layers). Fallback if joint image–text
attention is too heavy: condition on a single pooled image vector — still multimodal,
much lighter. The text-only path is identical to a standard Track-3 char-GPT, so the
project degrades gracefully to classic Track 3 with almost no wasted work.

## Allowed libraries (per assignment)

Python standard library, NumPy, Matplotlib, Seaborn, scikit-learn (preprocessing/
visualization only). Network implemented from scratch; tokenizer may be pre-built.
