import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

import llm

#################################
## CONSTANTS
#################################

DATASETS = [
    "imdb",
    "ag_news",
]


DATASET_CONFIGS = {
    "imdb": {
        "name": "imdb",
        "library": "datasets",
        "task": "classification",
        "text_field": "text",
        "label_field": "label",
    },
    "ag_news": {
        "name": "ag_news",
        "library": "datasets",
        "task": "classification",
        "text_field": "text",
        "label_field": "label",
    },
}


#################################
## Class definitions
#################################

class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
  
#################################
## Private Functions
#################################



#################################
## Public Functions
#################################

def get_dataset(name, tokenizer=None,batch_size=32, split="train"):
    if name not in DATASETS:
        raise ValueError(f"Dataset '{name}' not found. Available datasets: {DATASETS}")
    
    config = DATASET_CONFIGS[name]
    dataset = None
    if config["library"] == "datasets":
        dataset =  load_dataset(config["name"], split=split)
    encodings = tokenizer(dataset[config["text_field"]][:5000], truncation=True, padding=True, return_tensors="pt")

    if config["task"] == "classification":
        dataset = ClassificationDataset(encodings, dataset[config["label_field"]][:5000])

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

#################################
## Test function
#################################

if __name__ == "__main__":
    model_name = "albert"
    model_dict = llm.get_model(model_name)
    transformer = model_dict["tokenizer"]
    dataset = get_dataset("imdb", transformer)
    print(dataset)

