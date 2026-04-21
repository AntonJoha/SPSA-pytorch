import json
import math
import os
import re

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
MAX_LOSS_MULTIPLIER = 3.0

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
    batch_size=16,
    epochs=50,
    repeats=1,
    lr_options=None,
    scaling_options=None,
    noise_factor=0.0,
    optimizer_name="spsa",
    results_dir=DEFAULT_RESULTS_DIR,
    results_filename=DEFAULT_RESULTS_FILE,
    verbose=True,
):
    if lr_options is None:
        lr_options = [1e-3, 3e-4, 1e-4]
    if scaling_options is None:
        scaling_options = [1e-3, 3e-4, 1e-4]
    optimizer_name = optimizer_name.lower()
    if optimizer_name not in {"spsa", "sgd"}:
        raise ValueError("optimizer_name must be either 'spsa' or 'sgd'")

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
                model.train()

                # Freeze every parameter first.
                for p in model.parameters():
                    p.requires_grad_(False)

                # Unfreeze only a tiny, SPSA-feasible subset:
                # - last transformer block
                # - small MLM head transforms / biases
                # Keep tied output projection weights frozen (e.g. decoder.weight),
                # otherwise trainable dimensionality explodes and SPSA diverges.
                all_names = [name for name, _ in model.named_parameters()]
                layer_idxs = []
                for name in all_names:
                    m = re.search(r"(?:encoder|transformer)\.layer\.(\d+)\.", name)
                    if m:
                        layer_idxs.append(int(m.group(1)))
                last_layer_idx = max(layer_idxs) if layer_idxs else None

                for name, p in model.named_parameters():
                    if (
                        optimizer_name != "spsa"
                        and last_layer_idx is not None
                        and re.search(
                        rf"(?:encoder|transformer)\.layer\.{last_layer_idx}\.", name
                        )
                    ):
                        p.requires_grad_(True)
                        continue

                    # BERT MLM small head (exclude decoder.weight)
                    if name.startswith("cls.predictions.transform."):
                        p.requires_grad_(True)
                        continue
                    if name in {"cls.predictions.bias", "cls.predictions.decoder.bias"}:
                        p.requires_grad_(True)
                        continue

                    # DistilBERT MLM small head (exclude vocab_projector.weight)
                    if name.startswith(("vocab_transform.", "vocab_layer_norm.")):
                        p.requires_grad_(True)
                        continue
                    if name in {"vocab_projector.bias"}:
                        p.requires_grad_(True)
                        continue

                    # Classification heads (if used)
                    if name.endswith(("classifier.weight", "classifier.bias")):
                        p.requires_grad_(True)

                if verbose:
                    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    total = sum(p.numel() for p in model.parameters())
                    print(f"Trainable params: {trainable}/{total}")
                    print(f"Optimizer: {optimizer_name.upper()}")

                spsa_optimizer = None
                sgd_optimizer = None
                if optimizer_name == "spsa":
                    spsa_optimizer = SPSA(model, loss_fn=None, lr=lr, delta=scale, noise_factor=noise_factor)
                else:
                    sgd_optimizer = torch.optim.SGD(
                        [p for p in model.parameters() if p.requires_grad],
                        lr=lr,
                    )

                to_save = {
                    "optimizer": optimizer_name,
                    "lr": lr,
                    "scaling": scale,
                    "loss": [],
                    "training_succeeded": False,
                }
                initial_loss = None
                for epoch in range(epochs):
                    total_loss = 0.0
                    for batch in dataloader:
                        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                        if optimizer_name == "spsa":
                            batch_loss = spsa_optimizer.step(batch)
                        else:
                            sgd_optimizer.zero_grad(set_to_none=True)
                            batch_loss = model(**batch).loss
                            batch_loss.backward()
                            sgd_optimizer.step()
                        total_loss += batch_loss.item()

                    average_loss = total_loss / len(dataloader)
                    to_save["loss"].append(average_loss)
                    if initial_loss is None:
                        initial_loss = average_loss

                    if verbose and epoch % 2 == 0:
                        print(f"Repeat {repeat_idx + 1}, Epoch {epoch + 1}, Loss: {average_loss:.4f}")

                    if (
                        math.isnan(average_loss)
                        or (epoch > 0 and average_loss > MAX_LOSS_MULTIPLIER * initial_loss)
                    ):
                        break

                if len(to_save["loss"]) == 1:
                    to_save["training_succeeded"] = math.isfinite(to_save["loss"][0])
                elif len(to_save["loss"]) >= 2:
                    to_save["training_succeeded"] = (
                        math.isfinite(to_save["loss"][-1])
                        and to_save["loss"][-1] < to_save["loss"][0]
                    )
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
