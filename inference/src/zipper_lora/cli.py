"""Command-line inference for a released Zipper-LoRA checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LANGUAGE_TO_LID = {"en": 0, "fr": 1, "ru": 2, "th": 3, "de": 4, "es": 5,
                   "it": 6, "pt": 7, "ar": 8, "vi": 9, "ja": 10, "ko": 11}
VARIANTS = ("auto", "static", "hard", "soft", "independent", "flylora", "vanilla")
# Audio-encoder Linear layers adapted by every released method.
AUDIO_LORA_TARGETS = ["query", "value", "key", "out", "mlp.0", "mlp.2"]


def parse_args():
    parser = argparse.ArgumentParser(description="Transcribe one audio file with a released Zipper-LoRA checkpoint.")
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--language", required=True, choices=sorted(LANGUAGE_TO_LID))
    parser.add_argument("--variant", choices=VARIANTS, default="auto")
    parser.add_argument("--static-shared-rank", type=int)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--chunk-window", type=int, default=None,
                        help="Override the chunked encoder's chunk_n_window (default 400).")
    parser.add_argument("--no-chunk", action="store_true",
                        help="Force the non-chunked encoder even for a chunked checkpoint.")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def locate_checkpoint(root: Path, model_dir: Path | None = None):
    root = root.expanduser().resolve()
    if model_dir is not None:
        # Training-only checkpoints ship without *_hf/; reuse the base model directory.
        model_dirs = [model_dir.expanduser().resolve()]
    else:
        model_dirs = ([root] if (root / "config.json").is_file() else [])
        model_dirs += sorted(p for p in root.rglob("global_step*_hf") if (p / "config.json").is_file())
    states = sorted(root.rglob("mp_rank_00_model_states.pt"))
    if len(model_dirs) != 1 or len(states) != 1:
        raise RuntimeError(
            "--checkpoint-dir must identify one downloaded global_step directory containing "
            "global_step*_hf/ and mp_rank_00_model_states.pt. "
            f"Found model directories={model_dirs}, states={states}."
        )
    return model_dirs[0], states[0]


def resolve_device(torch, requested: str):
    available = {"cuda": torch.cuda.is_available(),
                 "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
                 "cpu": True}
    if requested == "auto":
        return next(item for item in ("cuda", "mps", "cpu") if available[item])
    if not available[requested]:
        raise RuntimeError(f"Requested device '{requested}' is unavailable in this PyTorch installation.")
    return requested


def infer_variant(requested: str, root: Path, state: dict):
    if requested != "auto":
        return requested
    keys = " ".join(state)
    if ".fly.A" in keys:
        return "flylora"
    if ".audio.base_model.model." in keys:
        return "vanilla"
    if "A_specific" in keys:
        return "independent"
    if ".lora.router." not in keys:
        return "static"
    # Hard and Soft share one layout; only the run name tells them apart.
    return "hard" if "hard" in str(root).lower() else "soft"


def find_llm_linears(model, nn):
    return [name for name, module in model.named_modules()
            if isinstance(module, nn.Linear) and "lm_head" not in name and "model.audio." not in name]


def _import_local(model_dir: Path, module: str):
    """Import ``module`` from ``model_dir`` treated as a Python package.

    HF's dynamic-module cache breaks on directory names containing dots
    (``qwen3-1.7b-...``) and does not copy nested relative imports such as
    ``audio.py -> audio_utils.py``, so the shipped code is imported in place.
    """
    import importlib
    import importlib.util
    import sys

    package = "zipper_lora_base_model"
    if package not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            package, model_dir / "__init__.py", submodule_search_locations=[str(model_dir)])
        sys.modules[package] = importlib.util.module_from_spec(spec)
    return importlib.import_module(f"{package}.{module}")


def load_audio_model(model_dir: Path, dtype, use_chunk_encoder: bool | None = None,
                     chunk_window: int | None = None):
    """Load the local Qwen-Audio class named in config.json ``architectures``.

    Base directories may lack ``auto_map`` (and its custom config drops it),
    so AutoModel would otherwise resolve to the text-only built-in Qwen3.
    ``use_chunk_encoder`` overrides ``audio_config.use_chunk_encoder`` so a
    single base directory serves both the chunked and ``-non-chunked`` released
    checkpoints; ``chunk_window`` overrides ``chunk_n_window``.
    """
    from transformers import AutoModelForCausalLM
    from transformers.cache_utils import Cache

    # The shipped generation code predates the get_max_length -> get_max_cache_shape rename.
    if not hasattr(Cache, "get_max_length") and hasattr(Cache, "get_max_cache_shape"):
        Cache.get_max_length = lambda self: self.get_max_cache_shape()

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    arch = (config.get("architectures") or [None])[0]
    if not (arch and (model_dir / "modeling_qwen3.py").is_file()):
        return AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=dtype)
    cls = getattr(_import_local(model_dir, "modeling_qwen3"), arch)
    if use_chunk_encoder is not None or chunk_window is not None:
        config_obj = cls.config_class.from_pretrained(str(model_dir))
        if use_chunk_encoder is not None:
            config_obj.audio_config.use_chunk_encoder = use_chunk_encoder
        if chunk_window is not None:
            config_obj.audio_config.chunk_n_window = chunk_window
        # Skip the class's override: it only copies mel_filters.npz next to the
        # module, which here is already the model directory itself.
        return super(cls, cls).from_pretrained(str(model_dir), config=config_obj, torch_dtype=dtype)
    return super(cls, cls).from_pretrained(str(model_dir), torch_dtype=dtype)


def load_audio_tokenizer(model_dir: Path):
    from transformers import AutoTokenizer

    config = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    ref = ((config.get("auto_map") or {}).get("AutoTokenizer") or [None])[0]
    if not ref:
        return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    module, name = ref.rsplit(".", 1)
    return getattr(_import_local(model_dir, module), name).from_pretrained(str(model_dir))


def run(args):
    """Execute inference from an argparse namespace.

    ``examples/inference.py`` owns the user-facing parser; this function keeps
    the loading implementation reusable by the optional console command.
    """
    if not args.audio.is_file():
        raise FileNotFoundError(args.audio)

    import torch
    import torch.nn as nn
    from peft import LoraConfig, TaskType, get_peft_model
    from .checkpoint import (inject_checkpoint_flylora, inject_checkpoint_zipper_lora, remap_static_moe_state,
                             set_checkpoint_language_ids, clear_checkpoint_language_ids)

    device = resolve_device(torch, args.device)
    dtype = torch.bfloat16 if device == "cuda" else (torch.float16 if device == "mps" else torch.float32)
    model_dir, state_path = locate_checkpoint(args.checkpoint_dir, getattr(args, "model_dir", None))
    # mmap avoids reading DeepSpeed's duplicated frozen fragments into memory.
    raw_state = torch.load(state_path, map_location="cpu", mmap=True)
    state = raw_state.get("module", raw_state) if isinstance(raw_state, dict) else raw_state
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format: {state_path}")
    variant = infer_variant(args.variant, args.checkpoint_dir, state)
    if variant == "static":
        state = remap_static_moe_state(state)

    # The base directory is the chunked variant; released run names without the
    # ``-non-chunked`` suffix were trained with the chunked encoder and reuse its
    # fixed eval window (400).  ``--no-chunk`` / ``--chunk-window`` override this.
    run_is_non_chunked = "non-chunked" in str(args.checkpoint_dir).lower()
    use_chunk_encoder = not run_is_non_chunked and not getattr(args, "no_chunk", False)
    chunk_window = getattr(args, "chunk_window", None)

    # The released *_hf directory ships its custom Qwen-Audio implementation;
    # trust_remote_code loads that local code only, never a remote revision.
    model = load_audio_model(model_dir, dtype, use_chunk_encoder=use_chunk_encoder,
                             chunk_window=chunk_window)
    if not hasattr(model, "model") or not hasattr(model.model, "audio"):
        raise RuntimeError("The downloaded model is not a supported Qwen-Audio Zipper-LoRA checkpoint.")
    # Audio special tokens extend the base vocabulary; match the trained table size.
    embed_key = next((k for k in state if k.endswith("embed_tokens.weight")), None)
    if embed_key is not None and state[embed_key].shape[0] != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(state[embed_key].shape[0], mean_resizing=False)
    if variant == "flylora":
        inject_checkpoint_flylora(model)
    elif variant == "vanilla":
        # Training wrapped the encoder in its own PeftModel (model.audio.base_model.model.*).
        model.model.audio = get_peft_model(model.model.audio, LoraConfig(
            r=32, lora_alpha=64, target_modules=AUDIO_LORA_TARGETS, lora_dropout=0.0, bias="none"))
    else:
        shared_embedding = nn.Embedding(len(LANGUAGE_TO_LID), 1280)
        shared_embedding.weight.requires_grad_(False)
        inject_checkpoint_zipper_lora(model, shared_embedding, variant=variant,
                                      static_shared_rank=args.static_shared_rank)
        # Static rank-MoE states carry no LID table (it routes by index only).
        if any(k.endswith("zipper_shared_embedding.weight") for k in state):
            model.model.audio.zipper_shared_embedding = shared_embedding
    model.to(device)
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=32, lora_alpha=64,
                                             target_modules=find_llm_linears(model, nn), lora_dropout=0.0, bias="none"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(state) == len(unexpected):
        raise RuntimeError("No checkpoint tensors matched the model. Check --variant and checkpoint directory.")
    print(f"Loaded {len(state) - len(unexpected)} tensors; {len(missing)} missing; {len(unexpected)} unexpected.")
    model.eval()

    tokenizer = load_audio_tokenizer(model_dir)
    eos = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    eos = tokenizer.eod_id if eos is None else eos
    tokenizer.pad_token_id = eos
    prompt = f"Transcribe the following {args.language} speech: <audio>{args.audio.resolve()}</audio>"
    if not hasattr(tokenizer, "process_audio"):
        raise RuntimeError("The checkpoint tokenizer does not expose process_audio; download the complete *_hf directory.")
    audio_info = tokenizer.process_audio(prompt, n_mels=128, audio_subsampling_scale=4)
    if audio_info == "ErrorAudio":
        raise RuntimeError(f"Unable to decode audio: {args.audio}")
    encoded = tokenizer([prompt], audio_info=[audio_info], return_tensors="pt", padding=True,
                        truncation=True, max_length=4096).to(device)
    set_checkpoint_language_ids(model, torch.tensor([LANGUAGE_TO_LID[args.language]], device=device))
    try:
        with torch.inference_mode():
            ids = model.generate(input_ids=encoded.input_ids, attention_mask=encoded.attention_mask,
                                 audio_info=[audio_info], max_new_tokens=args.max_new_tokens,
                                 eos_token_id=eos, pad_token_id=eos)
    finally:
        clear_checkpoint_language_ids(model)
    text = tokenizer.decode(ids[0], audio_info=audio_info, skip_special_tokens=False)
    text = text.split("</audio>", 1)[-1].replace("<|endoftext|>", "").strip()
    result = {"audio": str(args.audio.resolve()), "language": args.language, "variant": variant, "text": text}
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
