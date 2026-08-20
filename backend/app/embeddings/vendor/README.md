# Vendored third-party code

## `osnet.py`

OSNet model definition, copied verbatim from
[torchreid / deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid)
(`torchreid/reid/models/osnet.py`), MIT licensed — see `LICENSE-torchreid`.
Copyright (c) 2018 Kaiyang Zhou.

**Why vendored rather than depended on.** `pip install torchreid` works, but
its package `__init__` imports the entire training stack — dataset loaders,
training engines, TensorBoard — just to reach a model definition. That pulled
in `gdown` and `tensorboard` as undeclared import-time dependencies, each
failing in turn. The model file itself imports nothing beyond `torch`, so
copying it removes the whole fragile chain for 598 lines.

The file is unmodified so it can be diffed against upstream. Model weights are
**not** vendored; they download on first use into `data/models/` (see
`app/embeddings/reid.py`).

Reference: Zhou et al., *"Omni-Scale Feature Learning for Person
Re-Identification"*, ICCV 2019.
