#!/usr/bin/env python3
"""Run ASR with one released Zipper-LoRA checkpoint.

Example:
    python examples/inference.py \
      --checkpoint-dir checkpoints/ckpts/<run-name>/ckpts/global_stepXXXX \
      --audio /path/to/audio.wav --language en --variant soft

Install first with ``python -m pip install -e '.[inference]'``.  The command
discovers the ``global_step*_hf`` model directory and
``mp_rank_00_model_states.pt`` below ``--checkpoint-dir`` automatically.
"""

import argparse
from pathlib import Path

from zipper_lora.cli import LANGUAGE_TO_LID, VARIANTS, run


def parse_args():
    """All user-configurable inference arguments are defined here."""
    parser = argparse.ArgumentParser(
        description="Transcribe one audio file with a released Zipper-LoRA checkpoint."
    )
    parser.add_argument(
        "--checkpoint-dir", required=True, type=Path,
        help="Downloaded global_step directory containing *_hf/ and mp_rank_00_model_states.pt.",
    )
    parser.add_argument(
        "--model-dir", type=Path, default=None,
        help="Base HF model directory; use when --checkpoint-dir has no global_step*_hf/.",
    )
    parser.add_argument("--audio", required=True, type=Path, help="Input audio file.")
    parser.add_argument(
        "--language", required=True, choices=sorted(LANGUAGE_TO_LID),
        help="Language/expert code: " + ", ".join(LANGUAGE_TO_LID),
    )
    parser.add_argument(
        "--variant", choices=VARIANTS, default="auto",
        help="Audio adapter variant; auto detects it from the state keys "
             "(specify hard explicitly when the run name does not contain 'hard').",
    )
    parser.add_argument(
        "--static-shared-rank", type=int, default=None,
        help="Static Zipper-LoRA shared rank; defaults to r/2 (16 for r=32).",
    )
    parser.add_argument(
        "--device", choices=("auto", "cuda", "mps", "cpu"), default="auto",
        help="auto prefers CUDA, then Apple Silicon MPS, then CPU.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=500, help="Maximum generated transcript tokens.")
    parser.add_argument(
        "--chunk-window", type=int, default=None,
        help="Override the chunked encoder's chunk_n_window (default 400).",
    )
    parser.add_argument(
        "--no-chunk", action="store_true",
        help="Force the non-chunked encoder even for a chunked checkpoint.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON result path.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
