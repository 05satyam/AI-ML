"""
Run inference with a trained addition transformer.

Usage:
  python inference.py --a 12 --b 34
  python inference.py --prompt "27+15="
  python inference.py --interactive
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import BOS_ID, BOS_TOKEN, EOS_ID, decode, encode, split_prompt_and_answer
from model import AdditionTransformer, ModelConfig


def load_model(checkpoint_path: str, device: torch.device) -> tuple[AdditionTransformer, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_cfg = ModelConfig(**ckpt["model_config"])
    model = AdditionTransformer(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt.get("data_config", {})


def predict_sum(
    model: AdditionTransformer,
    a: int,
    b: int,
    device: torch.device,
    max_new_tokens: int = 8,
) -> dict:
    prompt = f"{BOS_TOKEN}{a}+{b}="
    prompt_ids = torch.tensor([encode(prompt)], device=device)

    with torch.no_grad():
        out_ids = model.generate(prompt_ids, max_new_tokens=max_new_tokens, eos_id=EOS_ID)

    full_text = decode(out_ids[0].tolist())
    _, predicted = split_prompt_and_answer(full_text)
    expected = str(a + b)

    return {
        "prompt": f"{a}+{b}=",
        "full_output": full_text,
        "predicted_sum": predicted,
        "expected_sum": expected,
        "correct": predicted == expected,
    }


def predict_from_prompt(
    model: AdditionTransformer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 8,
) -> str:
    if not prompt.startswith(BOS_TOKEN):
        prompt = BOS_TOKEN + prompt
    prompt_ids = torch.tensor([encode(prompt)], device=device)
    with torch.no_grad():
        out_ids = model.generate(prompt_ids, max_new_tokens=max_new_tokens, eos_id=EOS_ID)
    return decode(out_ids[0].tolist())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Addition transformer inference")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/addition_transformer.pt")
    parser.add_argument("--a", type=int, default=None)
    parser.add_argument("--b", type=int, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "mps"
        if args.device == "auto" and torch.backends.mps.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. Run `python train.py` first."
        )

    model, _ = load_model(str(ckpt_path), device)

    if args.interactive:
        print("Interactive mode. Type 'q' to quit.")
        while True:
            raw = input("\nEnter expression like 23+45 (no spaces): ").strip()
            if raw.lower() in {"q", "quit", "exit"}:
                break
            if "+" not in raw:
                print("Please use format: number+number")
                continue
            a_str, b_str = raw.split("+", 1)
            result = predict_sum(model, int(a_str), int(b_str), device)
            status = "OK" if result["correct"] else "WRONG"
            print(
                f"{result['prompt']} -> {result['predicted_sum']} "
                f"(expected {result['expected_sum']}) [{status}]"
            )
        return

    if args.prompt:
        text = predict_from_prompt(model, args.prompt, device)
        print(text)
        return

    if args.a is None or args.b is None:
        demos = [(3, 5), (7, 2), (9, 9), (4, 6), (0, 8)]
        print("No --a/--b provided. Running demo predictions:\n")
        for a, b in demos:
            result = predict_sum(model, a, b, device)
            mark = "✓" if result["correct"] else "✗"
            print(
                f"{mark} {result['prompt']} {result['predicted_sum']} "
                f"(expected {result['expected_sum']})"
            )
        return

    result = predict_sum(model, args.a, args.b, device)
    print(f"Prompt:   {result['prompt']}")
    print(f"Model:    {result['predicted_sum']}")
    print(f"Expected: {result['expected_sum']}")
    print(f"Correct:  {result['correct']}")


if __name__ == "__main__":
    main()
