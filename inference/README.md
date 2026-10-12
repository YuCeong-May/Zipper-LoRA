# Zipper-LoRA — Inference

Inference-only release of the Zipper-LoRA adapters described in
*Zipper-LoRA: Dynamic Parameter Decoupling for Speech-LLM based Multilingual
Speech Recognition*.  This directory ships the installable package, the
released-checkpoint ASR entry point, the Whisper LID-embedding export tool, and
the injection/checkpoint tests.  The training framework is intentionally not
included.

## Install

```bash
conda create -n zipper-lora python=3.11 -y
conda activate zipper-lora

git clone https://github.com/YuCeong-May/Zipper-LoRA.git
cd Zipper-LoRA/inference

# Apple Silicon / CPU
python -m pip install torch

# NVIDIA CUDA users: install the PyTorch wheel that matches the local CUDA
# runtime from https://pytorch.org/get-started/locally/ instead of the line above.

python -m pip install -e '.[inference]'
```

## Layout

```text
inference/
├── pyproject.toml                 # installable lightweight package
├── README.md
├── examples/
│   └── inference.py               # released-checkpoint ASR entry point
├── tools/
│   └── extract_whisper_lid_embeddings.py  # export Whisper language-token vectors
├── src/zipper_lora/
│   ├── __init__.py                # public API
│   ├── config.py                  # ZipperLoRAConfig
│   ├── layers.py                  # Static / Hard / Soft adapter modules
│   ├── inject.py                  # PEFT-style model injection/context API
│   ├── checkpoint.py              # compatible injection for published states
│   └── cli.py                     # `zipper-lora-asr` command
└── tests/
    └── test_injection.py          # shape and injection coverage
```

## Design

For a frozen linear layer `W0`, all variants add the same rank-`r` residual:

`y = W0 x + (alpha / r) B_merged(language) A x`.

`A` is a shared down-projection.  The variants differ only in how the
up-projection `B_merged` is built from `B_shared` and `B_specific[language]`:

| Variant | `B_merged` construction | Router |
| --- | --- | --- |
| `static` | `[B_shared[:, :r_s], B_specific[l][:, :r-r_s]]` | none; fixed rank split |
| `hard` | each rank column is selected from the shared or specific bank | LID -> LayerNorm -> Linear -> sigmoid -> threshold; STE while training |
| `soft` | each rank column is a continuous shared/specific mixture | LID -> LayerNorm -> Linear -> sigmoid |

The implementation names its router output `shared_gate`: `1` uses the
shared column and `0` uses the language-specific column.  This is equivalent
to the paper's Soft formulation after setting `shared_gate = 1 - p_specific`.
It makes Hard and Soft share one clear API and follows the semantics of the
existing inference module.

## Injection API

```python
import torch
from zipper_lora import (
    ZipperLoRAConfig,
    inject_zipper_lora,
    set_zipper_language_ids,
    clear_zipper_language_ids,
)

config = ZipperLoRAConfig(
    num_languages=12,
    lid_embedding_dim=1280,       # Whisper LID embedding dimension, or another LID encoder
    r=32,
    alpha=64,
    variant="soft",               # "static", "hard", or "soft"
    target_modules=("q_proj", "v_proj", "out", "mlp.0"),
    target_scope="model.audio",  # optional full-path scope
)

wrapped_paths = inject_zipper_lora(model, config)
set_zipper_language_ids(model, torch.tensor([language_id], device=device))
outputs = model(**inputs)
clear_zipper_language_ids(model)
```

`target_modules` accepts a leaf module name (`q_proj`) or a full module-path
suffix (`mlp.0`).  With no target list, it defaults to the common LLM
projections: `q/k/v/o_proj` and `gate/up/down_proj`.

Each adapter contains an LID embedding table.  It can be initialized with
Whisper language-token vectors exported by the bundled tool, or populated by
another application-owned LID encoder.

## Run a checkpoint

Two pieces are required and are hosted on Hugging Face (not in this repo):
the **base Qwen-Audio model** and one **trained checkpoint**.

1. Download the base model directory
   `qwen3-1.7b-chat-audio-whisper-v3-convproj-chunk` (its `modeling_qwen3.py`,
   `audio.py`, config, tokenizer and weights).  This supplies the model code,
   config and tokenizer used with `--model-dir`.
2. Download **one global-step directory** from the
   [Hugging Face `ckpts` collection](https://huggingface.co/YuCeong-May/Zipper-LoRA/tree/main/ckpts).
   Each contains `mp_rank_00_model_states.pt` (the full trained model state and
   its LID table).  A few steps also ship a `global_step*_hf/` model directory;
   for the others pass the base model directory with `--model-dir`.

```bash
# Base model (code + config + tokenizer + weights).
huggingface-cli download <base-model-repo-id> \
  --local-dir models/qwen3-1.7b-chat-audio-whisper-v3-convproj-chunk

# Select one <run-name> and one global step.
huggingface-cli download YuCeong-May/Zipper-LoRA \
  --include 'ckpts/<run-name>/ckpts/global_stepXXXX/**' \
  --local-dir checkpoints

python examples/inference.py \
  --checkpoint-dir checkpoints/ckpts/<run-name>/ckpts/global_stepXXXX \
  --model-dir models/qwen3-1.7b-chat-audio-whisper-v3-convproj-chunk \
  --audio /path/to/audio.wav \
  --language en
```

`--checkpoint-dir` automatically finds `mp_rank_00_model_states.pt` (and
`global_step*_hf` when `--model-dir` is omitted).  `--variant auto` (default)
detects the audio-adapter layout from the state keys:

| Run name | Variant | State layout |
| --- | --- | --- |
| `zipper-lora-soft*`, `zipper-lora-hard*` | `soft` / `hard` | `lora.{A,B_shared,B_specific,router}` |
| `zipper-lora-static*` | `static` | `moe.{A,B_shared,B_experts}` (LID-routed, shared rank 16) |
| `independent-lora*` | `independent` | `lora.{A_specific,B_specific}` |
| `flylora*` | `flylora` | `fly.{A,B,d}` (top-k = 8) |
| `vanilla-lora*` | `vanilla` | PEFT LoRA wrapping the audio encoder |

Hard and Soft share one layout, so auto selects Hard only when the checkpoint
path contains `hard`; otherwise pass `--variant hard` explicitly.  FlyLoRA and
vanilla LoRA are language-agnostic, but `--language` still sets the prompt.

The encoder path is chosen from the run name: checkpoints whose name contains
`-non-chunked` run the full-utterance Whisper encoder; the others run the
chunked encoder with a fixed window of 400 (the eval setting, `chunk_n_window`).
Override with `--chunk-window N` or force the non-chunked encoder with
`--no-chunk`.

The command auto-selects CUDA, then Apple Silicon MPS, then CPU.  To force a
backend use `--device cuda`, `--device mps`, or `--device cpu`.  CPU is useful
for a compatibility check but is generally impractical for a 1.7B model.

## Whisper LID embedding tool

`tools/extract_whisper_lid_embeddings.py` exports vectors from Whisper's
decoder language tokens.  The requested language order is preserved in the
saved tensor and therefore defines the Zipper-LoRA language-ID mapping.

```bash
python -m pip install -e '.[tools]'
python tools/extract_whisper_lid_embeddings.py \
  --model-id openai/whisper-large-v3 \
  --languages en fr ru th de es it pt ar vi ja ko \
  --output whisper_lid_embeddings.pt
```

To use the resulting tensor, copy it into every injected adapter before
inference.  The following helper makes this explicit:

```python
import torch
from zipper_lora import iter_zipper_layers

lid_embeddings = torch.load("whisper_lid_embeddings.pt", weights_only=True)
for _, layer in iter_zipper_layers(model):
    layer.zipper_lora.lid_embedding.weight.data.copy_(lid_embeddings)
```

## Install and check

```bash
python -m pip install pytest
python -m pytest
python examples/inference.py --help
```

## Citation

```bibtex
@article{ZipperLoRA2026,
  author={Mei, Yuxiang and Qiu, Delai and Liu, Shengping and Liang, Jiaen and Long, Yanhua},
  journal={IEEE Transactions on Audio, Speech and Language Processing},
  title={Zipper-LoRA: Dynamic Parameter Decoupling for Speech-LLM Based Multilingual Speech Recognition},
  year={2026},
  volume={34},
  pages={3956--3969},
  doi={10.1109/TASLPRO.2026.3719331}
}
```
