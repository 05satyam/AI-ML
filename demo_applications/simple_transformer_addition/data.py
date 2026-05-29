"""
Synthetic dataset for teaching a tiny transformer to add natural numbers.

We encode addition as a character sequence, e.g. "23+45=68".
The model learns next-token prediction (causal language modeling) on these strings.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

# Vocabulary: special tokens, digits, and operators.
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]
DIGIT_TOKENS = [str(d) for d in range(10)]
OPERATOR_TOKENS = ["+", "="]

VOCAB = SPECIAL_TOKENS + DIGIT_TOKENS + OPERATOR_TOKENS
STOI = {token: idx for idx, token in enumerate(VOCAB)}
ITOS = {idx: token for token, idx in STOI.items()}

PAD_ID = STOI[PAD_TOKEN]
BOS_ID = STOI[BOS_TOKEN]
EOS_ID = STOI[EOS_TOKEN]


def tokenize(text: str) -> list[str]:
    """
    Greedy longest-match tokenization.

    Special tokens like '<bos>' are multi-character, so we cannot split char-by-char.
    Real LLMs use BPE/SentencePiece — same idea: map text chunks to token IDs.
    """
    tokens: list[str] = []
    i = 0
    by_length = sorted(VOCAB, key=len, reverse=True)
    while i < len(text):
        for token in by_length:
            if text.startswith(token, i):
                tokens.append(token)
                i += len(token)
                break
        else:
            raise ValueError(f"Unknown text at index {i}: {text[i:]!r}")
    return tokens


def encode(text: str) -> list[int]:
    """Convert a string like '<bos>12+34=46<eos>' into token ids."""
    return [STOI[t] for t in tokenize(text)]


def decode(ids: list[int]) -> str:
    """Convert token ids back to a human-readable string."""
    return "".join(ITOS[i] for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID))


def format_example(a: int, b: int) -> str:
    """Build training text: '<bos>a+b=sum<eos>'."""
    return f"{BOS_TOKEN}{a}+{b}={a + b}{EOS_TOKEN}"


def split_prompt_and_answer(text: str) -> tuple[str, str]:
    """
    Split '12+34=46' into prompt '12+34=' and answer '46'.
    Works with or without BOS/EOS wrappers.
    """
    core = text.strip()
    if core.startswith(BOS_TOKEN):
        core = core[len(BOS_TOKEN) :]
    if core.endswith(EOS_TOKEN):
        core = core[: -len(EOS_TOKEN)]
    prompt, answer = core.split("=")
    return f"{prompt}=", answer


@dataclass
class DataConfig:
    max_number: int = 9
    max_seq_len: int = 16
    train_size: int = 5_000
    val_size: int = 500
    seed: int = 42


class AdditionDataset(Dataset):
    """Causal LM dataset: input = tokens[:-1], target = tokens[1:]."""

    def __init__(self, examples: list[str], max_seq_len: int):
        self.examples = examples
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        tokens = encode(self.examples[index])[: self.max_seq_len]
        if len(tokens) < 2:
            raise ValueError(f"Example too short: {self.examples[index]}")

        # Pad to fixed length so batches stack cleanly.
        pad_len = self.max_seq_len - len(tokens)
        tokens = tokens + [PAD_ID] * pad_len

        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)

        # Ignore loss on padded positions.
        loss_mask = (y != PAD_ID).float()
        return {"input_ids": x, "labels": y, "loss_mask": loss_mask}


def generate_examples(count: int, max_number: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    all_pairs = [
        (a, b)
        for a in range(max_number + 1)
        for b in range(max_number + 1)
    ]
    examples: list[str] = []
    if count <= len(all_pairs):
        rng.shuffle(all_pairs)
        for a, b in all_pairs[:count]:
            examples.append(format_example(a, b))
    else:
        # Oversample: repeat full grid, then fill with random pairs.
        repeats = count // len(all_pairs) + 1
        pool = all_pairs * repeats
        rng.shuffle(pool)
        for a, b in pool[:count]:
            examples.append(format_example(a, b))
    return examples


def build_datasets(config: DataConfig) -> tuple[AdditionDataset, AdditionDataset]:
    train_texts = generate_examples(config.train_size, config.max_number, config.seed)
    val_texts = generate_examples(config.val_size, config.max_number, config.seed + 1)
    train_ds = AdditionDataset(train_texts, config.max_seq_len)
    val_ds = AdditionDataset(val_texts, config.max_seq_len)
    return train_ds, val_ds


def demo_batch(config: DataConfig | None = None) -> None:
    config = config or DataConfig()
    train_ds, _ = build_datasets(config)
    sample = train_ds[0]
    tokens = sample["input_ids"].tolist()
    # Reconstruct one training example from (x, y) pairs.
    last_label = next(
        sample["labels"][i].item()
        for i in range(len(sample["labels"]) - 1, -1, -1)
        if sample["loss_mask"][i].item() > 0
    )
    text = decode([BOS_ID] + tokens + [last_label])
    print("Sample training string:", f"{BOS_TOKEN}{text}{EOS_TOKEN}")
    print("Token ids:", encode(format_example(12, 34)))
    print("Vocab size:", len(VOCAB))
    print("Vocab:", VOCAB)


if __name__ == "__main__":
    demo_batch()
