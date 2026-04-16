import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset


class ImdbMLMDataset(Dataset):
    def __init__(self, encodings, tokenizer, mlm_prob=0.15):
        self.encodings = encodings
        self.tokenizer = tokenizer
        self.mlm_prob = mlm_prob

    def __len__(self):
        return self.encodings["input_ids"].size(0)

    def __getitem__(self, idx):
        input_ids = self.encodings["input_ids"][idx].clone()
        attention_mask = self.encodings["attention_mask"][idx]
        labels = input_ids.clone()

        probability_matrix = torch.rand(input_ids.shape)
        special_tokens_mask = torch.tensor(
            self.tokenizer.get_special_tokens_mask(
                input_ids.tolist(), already_has_special_tokens=True
            ),
            dtype=torch.bool,
        )
        mask = (
            (probability_matrix < self.mlm_prob)
            & (attention_mask == 1)
            & (~special_tokens_mask)
        )
        labels[~mask] = -100

        # Standard BERT MLM corruption: 80% [MASK], 10% random, 10% unchanged.
        replace_prob = torch.rand(input_ids.shape)
        mask_token_positions = mask & (replace_prob < 0.8)
        random_token_positions = mask & (replace_prob >= 0.8) & (replace_prob < 0.9)

        input_ids[mask_token_positions] = self.tokenizer.mask_token_id
        random_words = torch.randint(
            low=0,
            high=self.tokenizer.vocab_size,
            size=input_ids.shape,
            dtype=torch.long,
        )
        input_ids[random_token_positions] = random_words[random_token_positions]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def get_imdb_mlm_dataloader(
    tokenizer,
    batch_size=8,
    truncated_dataset=300,
    max_length=128,
    mlm_prob=0.15,
):
    raw = load_dataset("imdb")["train"]
    texts = [t for t in raw["text"] if t and t.strip()]
    if truncated_dataset:
        texts = texts[:truncated_dataset]

    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    dataset = ImdbMLMDataset(encodings, tokenizer, mlm_prob=mlm_prob)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
