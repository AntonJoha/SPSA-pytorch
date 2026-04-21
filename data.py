from datasets import load_dataset

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
    },
    "ag_news": {
        "name": "ag_news",
        "library": "datasets",
    },
}


#################################
## Class definitions
#################################

  
#################################
## Private Functions
#################################

#################################
## Public Functions
#################################

def get_dataset(name, split="train"):
    if name not in DATASETS:
        raise ValueError(f"Dataset '{name}' not found. Available datasets: {DATASETS}")
    
    config = DATASET_CONFIGS[name]
    
    if config["library"] == "datasets":
        return load_dataset(config["name"], split=split)

#################################
## Test function
#################################

if __name__ == "__main__":

    dataset = get_dataset("imdb")
    print(dataset)

