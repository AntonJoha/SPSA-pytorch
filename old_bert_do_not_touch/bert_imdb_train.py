import torch

import llm
from imdb_dataset import get_imdb_mlm_dataloader
from spsa import SPSA


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _freeze_for_spsa_bert_mlm(model):
    """Freeze all parameters except the small MLM prediction head.

    Keeping trainable parameters under ~1 M is the practical threshold where
    SPSA gradient estimates have a reasonable signal-to-noise ratio.  The MLM
    head (``cls.predictions.transform.*`` + biases) is ~600 K parameters for
    bert-base-uncased.
    """
    for p in model.parameters():
        p.requires_grad_(False)

    # Small MLM head only (keep decoder.weight frozen due to tied embeddings).
    for name, p in model.named_parameters():
        if name.startswith("cls.predictions.transform."):
            p.requires_grad_(True)
        if name in {"cls.predictions.bias", "cls.predictions.decoder.bias"}:
            p.requires_grad_(True)

        # DistilBERT equivalent head (exclude vocab_projector.weight)
        if name.startswith(("vocab_transform.", "vocab_layer_norm.")):
            p.requires_grad_(True)
        if name in {"vocab_projector.bias"}:
            p.requires_grad_(True)


def run_simple_bert_imdb(
    batch_size=8,
    epochs=10,
    lr=3e-4,
    delta=1e-3,
    noise_factor=0.0,
    num_estimates=4,
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
    if verbose:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {trainable:,} / {total:,}")
    optimizer = SPSA(model, loss_fn=None, lr=lr, delta=delta, noise_factor=noise_factor, num_estimates=num_estimates)

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
