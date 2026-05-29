"""
Train the addition transformer on synthetic examples.

Usage:
  python train.py
  python train.py --epochs 15 --max-number 999
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import BOS_ID, DataConfig, EOS_ID, build_datasets, decode, encode
from model import AdditionTransformer, ModelConfig, count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny addition transformer")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-number", type=int, default=9)
    parser.add_argument("--train-size", type=int, default=10_000)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--save-path", type=str, default="checkpoints/addition_transformer.pt")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction="none",
    )
    loss = loss.view(labels.shape)
    return (loss * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)


def evaluate(model: AdditionTransformer, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0.0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            key_padding_mask = input_ids == 0  # PAD_ID

            logits = model(input_ids, key_padding_mask=key_padding_mask)
            loss = masked_cross_entropy(logits, labels, loss_mask)
            total_loss += loss.item() * loss_mask.sum().item()
            total_tokens += loss_mask.sum().item()
    return total_loss / max(total_tokens, 1.0)


def exact_match_accuracy(
    model: AdditionTransformer,
    device: torch.device,
    max_number: int,
    samples: int = 200,
) -> float:
    """Fraction of random prompts where generated answer digits exactly match a+b."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for _ in range(samples):
            a = torch.randint(0, max_number + 1, (1,)).item()
            b = torch.randint(0, max_number + 1, (1,)).item()
            prompt = f"<bos>{a}+{b}="
            prompt_ids = torch.tensor([encode(prompt)], device=device)
            out_ids = model.generate(prompt_ids, max_new_tokens=8, eos_id=EOS_ID)
            text = decode(out_ids[0].tolist())
            expected = str(a + b)
            if "=" in text:
                predicted = text.split("=")[-1]
                if predicted == expected:
                    correct += 1
    return correct / samples


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
    print(f"Using device: {device}")

    data_cfg = DataConfig(
        max_number=args.max_number,
        train_size=args.train_size,
        val_size=args.val_size,
    )
    train_ds, val_ds = build_datasets(data_cfg)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model_cfg = ModelConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=data_cfg.max_seq_len,
    )
    model = AdditionTransformer(model_cfg).to(device)
    print(f"Trainable parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = float("inf")
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_tokens = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            key_padding_mask = input_ids == 0

            logits = model(input_ids, key_padding_mask=key_padding_mask)
            loss = masked_cross_entropy(logits, labels, loss_mask)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item() * loss_mask.sum().item()
            running_tokens += loss_mask.sum().item()
            progress.set_postfix(train_loss=running_loss / running_tokens)

        val_loss = evaluate(model, val_loader, device)
        em_acc = exact_match_accuracy(model, device, args.max_number)
        print(
            f"Epoch {epoch}: val_loss={val_loss:.4f} | exact_match_acc={em_acc:.1%}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model_cfg.__dict__,
                    "data_config": data_cfg.__dict__,
                },
                save_path,
            )
            print(f"  Saved checkpoint -> {save_path}")

    print("\nDone. Try inference:")
    print("  python inference.py --a 27 --b 15")


if __name__ == "__main__":
    main()
