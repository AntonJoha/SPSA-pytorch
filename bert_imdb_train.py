import torch
import re

import llm
from imdb_dataset import get_imdb_mlm_dataloader
from spsa import SPSA


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _freeze_for_spsa_bert_mlm(model):
    for p in model.parameters():
        p.requires_grad_(False)

    # Last encoder/transformer block (detected dynamically)
    layer_idxs = []
    for name, _ in model.named_parameters():
        m = re.search(r"(?:encoder|transformer)\.layer\.(\d+)\.", name)
        if m:
            layer_idxs.append(int(m.group(1)))
    last_layer_idx = max(layer_idxs) if layer_idxs else None

    for name, p in model.named_parameters():
        if last_layer_idx is not None and re.search(
            rf"(?:encoder|transformer)\.layer\.{last_layer_idx}\.", name
        ):
            p.requires_grad_(True)

    # Small MLM head only (keep decoder.weight frozen due tied embeddings)
    for name, p in model.named_parameters():
        if name.startswith("cls.predictions.transform."):
            p.requires_grad_(True)
        if name in {"cls.predictions.bias", "cls.predictions.decoder.bias"}:
            p.requires_grad_(True)


def run_simple_bert_imdb(
    batch_size=8,
    epochs=10,
    lr=3e-4,
    delta=1e-3,
    noise_factor=0.0,
    truncated_dataset=300,
    verbose=True,
):
    model_dict = llm.get_model("bert-base-uncased")
    tokenizer = model_dict["tokenizer"]
    model = model_dict["model"].to(device)
    model.train()

    dataloader = get_imdb_mlm_dataloader(
        tokenizer=tokenizer,
        batch_size=batch_size,
        truncated_dataset=truncated_dataset,
    )

    _freeze_for_spsa_bert_mlm(model)
    optimizer = SPSA(model, loss_fn=None, lr=lr, delta=delta, noise_factor=noise_factor)

    losses = []
    for epoch in range(epochs):
        total_loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                total_loss += optimizer.step(batch).item()

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f}")

    return losses


if __name__ == "__main__":
    run_simple_bert_imdb()
