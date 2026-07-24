Subject: Final project topic — image captioning Transformer from scratch (Track 3)

Dear Prof. Bruni,

Our group would like to propose a topic for the final project and check that it fits
Track 3 before we start.

**Idea.** A decoder-only Transformer (nanoGPT-style) that generates an image caption
word-by-word, implemented from scratch in NumPy. It keeps the full Track-3 architecture —
attention, LayerNorm, embeddings, backprop, Adam — and adds one multimodal twist: the
model conditions on an image (encoded as a short sequence of patch vectors) prepended to
the caption tokens. Captioning then becomes next-token language modeling conditioned on
an image, so it stays within Track 3 while connecting Computer Vision and NLP.

**Why this topic.** We are genuinely interested in both CV and NLP, and this is the
natural point where they meet — grounding language in what the model sees. We also know
this direction is close to your research, and we would be glad to hear your thoughts.

**Scope.** Deliberately small (~1–3M parameters, 3–4 layers), trained on a subset of
Flickr8k (real photos + human captions) with downscaled images so it runs on CPU. We plan
to gradient-check every module, show training/validation loss curves, generate captions on
held-out images, and — as our analysis — overlay attention weights on the image to see
whether words like "dog", "red", or "left" attend to the right regions.

**Questions.**
1. Is this multimodal extension of Track 3 acceptable for the assignment?
2. Do you have any input on the grounding/attention analysis, or any datasets you would
   prefer we use?

Thank you very much — we would appreciate your feedback.

Best regards,
Mustafa
[on behalf of the group]
