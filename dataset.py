import torch
from datasets import load_dataset
from torch.utils.data import TensorDataset, DataLoader, Dataset

#################################
## CONSTANTS
#################################
DATASETS = [
    "imdb"
]

DATASET_TYPE = [
    "MLM"
]

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
            "labels": labels
        }

#################################
## Functions
#################################



def get_dataset(name, dataset_type, tokenizer, batch_size):
    data = None
    if name == "imdb":
        data = load_dataset(name)["train"]["text"][:]
    
        
    encoding = tokenizer(
        data,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )
    dataset = None
    if dataset_type == "MLM":
        dataset = MLMDataset(encoding, tokenizer)

    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=True)
    return dataloader


#################################
## Test function
#################################

if __name__ == "__main__":
    print("Test?")