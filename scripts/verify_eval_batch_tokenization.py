from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    conversations = [
        [{"role": "user", "content": "Compute 2 + 2."}],
        [
            {
                "role": "user",
                "content": "Find the positive integer n such that n squared equals 81.",
            }
        ],
    ]
    batch = tokenizer.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        padding=True,
        return_tensors="pt",
        return_dict=True,
    )
    individual_ids = [
        tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
        )
        for conversation in conversations
    ]
    for index, expected in enumerate(individual_ids):
        actual = batch["input_ids"][index][batch["attention_mask"][index].bool()].tolist()
        if actual != expected:
            raise RuntimeError(f"Batched chat-template tokens differ for conversation {index}")
    print(
        json.dumps(
            {
                "status": "passed",
                "padding_side": tokenizer.padding_side,
                "batch_shape": list(batch["input_ids"].shape),
                "individual_lengths": [len(token_ids) for token_ids in individual_ids],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
