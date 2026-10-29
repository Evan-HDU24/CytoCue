import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .batching import context_batch, move_context


class CellBatches(Dataset):
    def __init__(self, dataset, batch_size, indices=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices = np.arange(len(dataset)) if indices is None else np.asarray(indices)

    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)

    def __getitem__(self, index):
        ids = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        return context_batch(self.dataset, ids.tolist())


def batches(dataset, batch_size, workers, seed, indices=None):
    return DataLoader(
        CellBatches(dataset, batch_size, indices),
        batch_size=None,
        num_workers=workers,
        pin_memory=True,
        generator=torch.Generator().manual_seed(seed),
    )


def autocast(device, precision):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and precision == "bf16",
    )


@torch.inference_mode()
def predict(model, dataset, options):
    device = next(model.parameters()).device
    probabilities = np.empty(len(dataset), dtype=np.float32)
    model.eval()
    for step, batch in enumerate(
        batches(dataset, options["batch_size"], options["workers"], options["seed"])
    ):
        with autocast(device, options["precision"]):
            features = model.encoder(move_context(batch, device)).float()
        probabilities[batch["index"].numpy()] = model.readout(features).sigmoid().cpu().numpy()
        if (step + 1) % 100 == 0:
            print(
                f"Predicted {min((step + 1) * options['batch_size'], len(dataset))}/{len(dataset)} cells",
                flush=True,
            )
    if not np.isfinite(probabilities).all():
        raise ValueError("Nonfinite prediction")
    records = [
        {
            "image": dataset.pairs[row[0]]["unit"],
            "block": dataset.pairs[row[0]]["block"],
            "marker": dataset.pairs[row[0]]["marker"],
            "cell_id": row[1],
            "target": row[2],
            "probability": float(p),
        }
        for row, p in zip(dataset.rows, probabilities)
    ]
    dataset.close()
    return pd.DataFrame(records)


@torch.inference_mode()
def extract_features(encoder, dataset, options, path):
    device = next(encoder.parameters()).device
    matrix = np.lib.format.open_memmap(
        Path(path), mode="w+", dtype=np.float32, shape=(len(dataset), encoder.output_dimension)
    )
    encoder.eval()
    for batch in batches(dataset, options["batch_size"], options["workers"], options["seed"]):
        with autocast(device, options["precision"]):
            values = encoder(move_context(batch, device)).float().cpu().numpy()
        matrix[batch["index"].numpy()] = values
    matrix.flush()
    dataset.close()
    return matrix
