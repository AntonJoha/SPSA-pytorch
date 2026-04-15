import json
import math
import os

import torch

import dataset
import llm
from spsa import SPSA

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#################################
## CONSTANTS
#################################

DEFAULT_RESULTS_DIR = "results"
DEFAULT_RESULTS_FILE = "spsa_full.json"

#################################
## Class definitions
#################################


#################################
## Functions
#################################


def _to_device(batch):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def run_spsa_experiments(
    model_name="distilbert-base-uncased",
    dataset_name="ag_news",
    dataset_type="MLM",
    batch_size=3,
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
        lr_options = [1e-6, 1e-7, 1e-5]
    if scaling_options is None:
        scaling_options = [1e-4, 1e-5, 1e-6]

    model_dict = llm.get_model(model_name)
    dataloader = dataset.get_dataset(dataset_name, dataset_type, model_dict["tokenizer"], batch_size)

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
                model.eval()
                spsa_optimizer = SPSA(model, loss_fn=None, lr=lr, delta=scale, noise_factor=noise_factor)

                to_save = {"lr": lr, "scaling": scale, "loss": []}
                for epoch in range(epochs):
                    total_loss = 0.0
                    with torch.no_grad():
                        for batch in dataloader:
                            batch = _to_device(batch)
                            batch_loss = spsa_optimizer.step_with_closure(lambda b=batch: model(**b).loss)
                            total_loss += batch_loss.item()

                    average_loss = total_loss / len(dataloader)
                    to_save["loss"].append(average_loss)

                    if verbose and epoch % 2 == 0:
                        print(f"Repeat {repeat_idx + 1}, Epoch {epoch + 1}, Loss: {average_loss:.4f}")

                    if math.isnan(total_loss) or average_loss > 20:
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
