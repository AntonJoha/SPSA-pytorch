import torch
from datasets import load_dataset
from torch.utils.data import TensorDataset, DataLoader, Dataset

#################################
## CONSTANTS
#################################
# Supported dataset names
DATASETS = [
    "imdb",
    "ag_news",
    "yelp_review_full",
    "sst2",
    "wikitext",
]

# Supported task types
DATASET_TYPE = [
    "MLM",           # Masked Language Modelling (e.g. BERT pre-training)
    "classification", # Supervised sequence classification
    "CLM",           # Causal Language Modelling (e.g. GPT pre-training)
]

# Per-dataset configuration: (hf_name, hf_config, split, text_field, label_field)
# label_field is None for unsupervised (MLM / CLM) tasks.
_DATASET_CONFIG = {
    "imdb":             ("imdb",             None,                  "train", "text",     "label"),
    "ag_news":          ("ag_news",           None,                  "train", "text",     "label"),
    "yelp_review_full": ("yelp_review_full",  None,                  "train", "text",     "label"),
    "sst2":             ("glue",              "sst2",                "train", "sentence", "label"),
    "wikitext":         ("wikitext",          "wikitext-2-raw-v1",   "train", "text",     None),
}

#################################
## Class definitions
#################################

class MLMDataset(Dataset):
    def __init__(self, encodings, tokenizer, mlm_prob=0.3):
        self.encodings = encodings
        self.tokenizer = tokenizer
        self.mlm_prob = mlm_prob

    def __len__(self):
        return self.encodings["input_ids"].size(0)

    def __getitem__(self, idx):
        input_ids = self.encodings["input_ids"][idx].clone()
        labels = input_ids.clone()

        # Mask tokens
        probability_matrix = torch.rand(input_ids.shape)
        mask = probability_matrix < self.mlm_prob
        input_ids[mask] = self.tokenizer.mask_token_id

        return {
            "input_ids": input_ids,
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": labels,
        }


class ClassificationDataset(Dataset):
    """Dataset for supervised sequence-classification tasks."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return self.encodings["input_ids"].size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


class CLMDataset(Dataset):
    """Dataset for causal language-modelling tasks (next-token prediction)."""

    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return self.encodings["input_ids"].size(0)

    def __getitem__(self, idx):
        input_ids = self.encodings["input_ids"][idx]
        attention_mask = self.encodings["attention_mask"][idx]
        # Labels are the same as input_ids; loss is computed on shifted tokens
        # inside the model (standard CLM convention). Padding positions must
        # be ignored by the loss.
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


#################################
## Functions
#################################


def get_dataset(name, dataset_type, tokenizer, batch_size):
    """Load *name* dataset, tokenize it, and return a DataLoader.

    Parameters
    ----------
    name : str
        One of the keys in DATASETS / _DATASET_CONFIG.
    dataset_type : str
        One of "MLM", "classification", or "CLM".
    tokenizer : transformers tokenizer
        A HuggingFace tokenizer compatible with the chosen model.
    batch_size : int
        Batch size for the returned DataLoader.
    """
    if name not in _DATASET_CONFIG:
        raise ValueError(f"Unsupported dataset '{name}'. Choose from: {DATASETS}")
    if dataset_type not in DATASET_TYPE:
        raise ValueError(f"Unsupported dataset_type '{dataset_type}'. Choose from: {DATASET_TYPE}")

    hf_name, hf_config, split, text_field, label_field = _DATASET_CONFIG[name]

    if hf_config is not None:
        raw = load_dataset(hf_name, hf_config)[split]
    else:
        raw = load_dataset(hf_name)[split]

    # Filter out empty strings (common in wikitext) in a single pass so that
    # texts and labels remain aligned.
    if label_field is not None:
        filtered = [(row[text_field], row[label_field]) for row in raw if row[text_field] and row[text_field].strip()]
        texts, labels_raw = zip(*filtered) if filtered else ([], [])
        texts = list(texts)
    else:
        texts = [t for t in raw[text_field] if t and t.strip()]
        labels_raw = []

    encoding = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )

    if dataset_type == "MLM":
        dataset = MLMDataset(encoding, tokenizer)
    elif dataset_type == "CLM":
        dataset = CLMDataset(encoding)
    elif dataset_type == "classification":
        if label_field is None:
            raise ValueError(
                f"Dataset '{name}' does not provide labels; "
                "use 'MLM' or 'CLM' as dataset_type."
            )
        labels = torch.tensor(list(labels_raw), dtype=torch.long)
        dataset = ClassificationDataset(encoding, labels)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


#################################
## Test function
#################################

if __name__ == "__main__":
    print("Test?")