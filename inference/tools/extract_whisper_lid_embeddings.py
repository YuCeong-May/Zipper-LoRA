#!/usr/bin/env python3
"""Export Whisper language-token embeddings for Zipper-LoRA routing.

The output is a ``torch.Tensor`` of shape ``[num_languages, hidden_size]``.
Its row order exactly follows ``--languages`` and must match the language-ID
mapping used when calling ``set_zipper_language_ids``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


DEFAULT_LANGUAGES = ("en", "fr", "ru", "th", "de", "es", "it", "pt", "ar", "vi", "ja", "ko")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        default="openai/whisper-large-v3",
        help="Hugging Face model ID or local Whisper model directory.",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=list(DEFAULT_LANGUAGES),
        help="ISO 639-1 language codes, in the desired Zipper-LoRA ID order.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("whisper_lid_embeddings.pt"),
        help="Output tensor path.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalize each language vector before saving.",
    )
    return parser.parse_args()


def extract_lid_embeddings(
    model_id: str,
    languages: list[str],
    normalize: bool = False,
) -> torch.Tensor:
    """Return decoder language-token vectors in exactly the supplied order."""
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    tokenizer = processor.tokenizer
    embedding_weight = model.model.decoder.embed_tokens.weight.detach().cpu()

    vectors: list[torch.Tensor] = []
    missing: list[str] = []
    for language in languages:
        token = f"<|{language}|>"
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == tokenizer.unk_token_id:
            missing.append(f"{language} ({token})")
            continue
        vectors.append(embedding_weight[token_id])
        print(f"{len(vectors) - 1:>3}: {language:<8} token_id={token_id}")

    if missing:
        raise ValueError("Language tokens not found: " + ", ".join(missing))
    if not vectors:
        raise ValueError("No language embeddings were extracted.")

    result = torch.stack(vectors)
    if normalize:
        result = torch.nn.functional.normalize(result, p=2, dim=-1)
    return result


def main() -> None:
    args = parse_args()
    embeddings = extract_lid_embeddings(args.model_id, args.languages, args.normalize)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, args.output)
    print(f"Saved {tuple(embeddings.shape)} tensor to {args.output}")


if __name__ == "__main__":
    main()
