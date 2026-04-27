#!/usr/bin/env python3
"""Gemma 4 e4b — local chat with optional image input.

Usage:
  gemma "your prompt"                    # text one-shot
  gemma                                   # interactive text chat
  gemma -i path/to/img.jpg "describe"    # image + text one-shot
"""
import argparse
import sys
from pathlib import Path

REPO = "mlx-community/gemma-4-e4b-it-4bit"


def parse_args():
    p = argparse.ArgumentParser(prog="gemma", add_help=True)
    p.add_argument("-i", "--image", action="append", default=[],
                   help="Path to image (can be repeated)")
    p.add_argument("-a", "--audio", action="append", default=[],
                   help="Path to audio file (can be repeated)")
    p.add_argument("prompt", nargs="*", help="Prompt text")
    return p.parse_args()


def run_multimodal(args):
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    print("Loading gemma-4-e4b (multimodal)…", end=" ", flush=True)
    model, processor = load(REPO)
    config = load_config(REPO)
    print("ready.\n", flush=True)

    prompt_text = " ".join(args.prompt) or "Describe this."
    formatted = apply_chat_template(
        processor, config, prompt_text,
        num_images=len(args.image), num_audios=len(args.audio),
    )
    out = generate(
        model, processor, formatted,
        image=args.image or None,
        audio=args.audio or None,
        max_tokens=-1, verbose=True,
    )
    return out


def run_text(args):
    from mlx_lm.utils import load_model, load_tokenizer, snapshot_download
    from mlx_lm.generate import stream_generate

    print("Loading gemma-4-e4b…", end=" ", flush=True)
    path = Path(snapshot_download(REPO))
    model, config = load_model(path, strict=False)
    tok = load_tokenizer(path, eos_token_ids=config.get("eos_token_id"))
    print("ready.\n", flush=True)

    history = []

    def turn(user_msg: str):
        history.append({"role": "user", "content": user_msg})
        prompt = tok.apply_chat_template(history, add_generation_prompt=True)
        out = ""
        for resp in stream_generate(model, tok, prompt=prompt, max_tokens=-1):
            print(resp.text, end="", flush=True)
            out += resp.text
        print()
        history.append({"role": "assistant", "content": out})

    if args.prompt:
        turn(" ".join(args.prompt))
        return

    print("Interactive — Ctrl-D to quit, /reset to clear history.\n")
    while True:
        try:
            line = input("» ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line == "/reset":
            history.clear()
            print("[history cleared]")
            continue
        turn(line)


def main():
    args = parse_args()
    if args.image or args.audio:
        run_multimodal(args)
    else:
        run_text(args)


if __name__ == "__main__":
    main()
