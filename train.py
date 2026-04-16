import json
import math
import os

import torch

import dataset
import llm
from spsa import SPSA

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
#################################
## CONSTANTS
#################################

DEFAULT_RESULTS_DIR = "results"
DEFAULT_RESULTS_FILE = "spsa_full.json"
MAX_STABLE_LOSS = 20.0

#################################
## Class definitions
#################################


#################################
## Functions
#################################

def run_spsa_experiments(
    model_name="bert-base-uncased",
    dataset_name="imdb",
    dataset_type="MLM",
    batch_size=10,
    epochs=50,
    repeats=10,
    lr_options=None,
    scaling_options=None,
    noise_factor=0.0,
    results_dir=DEFAULT_RESULTS_DIR,
    results_filename=DEFAULT_RESULTS_FILE,
    verbose=True,
):
    if lr_options is None:
        lr_options = [1e-1,1e-2,1e-3,1e-4, 1e-5, 1e-6,1e-7]
    if scaling_options is None:
        scaling_options = [1e-1,1e-2,1e-3,1e-4, 1e-5, 1e-6,1e-7]

    model_dict = llm.get_model(model_name)
    dataloader = dataset.get_dataset(dataset_name, dataset_type, model_dict["tokenizer"], batch_size, truncated_dataset=300)

    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, results_filename)

    try:
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    for repeat_idx in range(repeats):
        for lr in lr_options:
            for scale in scaling_options:
                if verbose:
                    print("--------------------START---------------------------------")
                    print("LR", lr, "Scale", scale)

                model = llm.get_model(model_name)["model"].to(device)
                print("HERE?")
                model.eval()
                spsa_optimizer = SPSA(model, loss_fn=None, lr=lr, delta=scale, noise_factor=noise_factor)

                to_save = {"lr": lr, "scaling": scale, "loss": []}
                for epoch in range(epochs):
                    total_loss = 0.0
                    with torch.no_grad():
                        for batch in dataloader:
                            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                            

                            batch_loss = spsa_optimizer.step(batch)
                            total_loss += batch_loss.item()

                    average_loss = total_loss / len(dataloader)
                    to_save["loss"].append(average_loss)

                    if verbose and epoch % 2 == 0:
                        print(f"Repeat {repeat_idx + 1}, Epoch {epoch + 1}, Loss: {average_loss:.4f}")

                    if math.isnan(average_loss) or average_loss > MAX_STABLE_LOSS:
                        break

                data.append(to_save)
                with open(results_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)

                if verbose:
                    print("-----------------------------END---------------------")

    return data, results_path


#################################
## Test function
#################################

if __name__ == "__main__":
    print("Training with SPSA...")
    run_spsa_experiments()
