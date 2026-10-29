from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .batching import move_context
from .inference import autocast, batches, extract_features
from .model import CytoCue
from .readout import ContextualGatedReadout


def loss_function(logits, target):
    logits, target = logits.float(), target.float()
    probability = logits.sigmoid()
    dice = 1 - (2 * (probability * target).sum() + 1) / (probability.sum() + target.sum() + 1)
    return 0.5 * F.binary_cross_entropy_with_logits(logits, target) + 0.5 * dice


def sample_indices(dataset, count, seed):
    groups = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(dataset.rows):
        pair = dataset.pairs[row[0]]
        if pair["split"] != "train" or row[2] not in (0, 1):
            raise ValueError("Training requires annotated training cells")
        groups[(pair["resource"], pair["source"])][pair["marker"]].append(index)
    rng = np.random.default_rng(seed)
    sources = sorted(groups)
    result = []
    for _ in range(count):
        source = sources[int(rng.integers(len(sources)))]
        markers = sorted(groups[source])
        marker = markers[int(rng.integers(len(markers)))]
        rows = groups[source][marker]
        result.append(rows[int(rng.integers(len(rows)))])
    return np.asarray(result)


def augment(batch):
    count = len(batch["fine"])
    device = batch["fine"].device
    turns = torch.randint(4, (count, 1), device=device)
    flips = torch.randint(2, (count, 2), device=device)
    for size in sorted({batch["fine"].shape[-1], batch["context"].shape[-1]}):
        y, x = torch.meshgrid(
            torch.arange(size, device=device), torch.arange(size, device=device), indexing="ij"
        )
        y, x = y.reshape(1, -1), x.reshape(1, -1)
        yy = torch.where(flips[:, 0:1].bool(), size - 1 - y, y)
        xx = torch.where(flips[:, 1:2].bool(), size - 1 - x, x)
        sy = torch.where(
            turns == 0,
            yy,
            torch.where(turns == 1, xx, torch.where(turns == 2, size - 1 - yy, size - 1 - xx)),
        )
        sx = torch.where(
            turns == 0,
            xx,
            torch.where(turns == 1, size - 1 - yy, torch.where(turns == 2, size - 1 - xx, yy)),
        )
        indices = (sy * size + sx)[:, None]
        for key, value in list(batch.items()):
            if value.shape[-1] == size:
                batch[key] = (
                    value.flatten(2)
                    .gather(2, indices.expand(-1, value.shape[1], -1))
                    .reshape_as(value)
                )
    return batch


def train(dataset, options):
    device = torch.device(options["device"])
    if device.type != "cuda":
        raise ValueError("Training requires a CUDA device")
    settings = options["training"]
    output = Path(options["output"])
    if (output / "cytocue.pt").exists():
        raise FileExistsError("Choose an unused output folder to preserve the existing model")
    model = CytoCue(**options["architecture"], pretrained=True).to(
        device, memory_format=torch.channels_last
    )
    representation_head = nn.Linear(model.encoder.output_dimension, 1).to(device)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(model.encoder.cell_encoder.parameters())
                + list(model.encoder.context_encoder.parameters()),
                "lr": settings["encoder_learning_rate"],
            },
            {
                "params": list(model.encoder.context_projection.parameters())
                + list(representation_head.parameters()),
                "lr": settings["projection_learning_rate"],
            },
        ],
        weight_decay=settings["weight_decay"],
    )
    for epoch in range(settings["encoder_epochs"]):
        model.train()
        model.encoder.freeze_backbones(epoch < settings["warmup_epochs"])
        indices = sample_indices(
            dataset,
            settings["encoder_steps"] * settings["encoder_batch_size"],
            options["seed"] + epoch,
        )
        total = 0.0
        for batch in batches(
            dataset,
            settings["encoder_batch_size"],
            options["workers"],
            options["seed"] + epoch,
            indices,
        ):
            optimizer.zero_grad(set_to_none=True)
            with autocast(device, options["precision"]):
                features = model.encoder(augment(move_context(batch, device)))
            logits = representation_head(features.float()).squeeze(-1)
            loss = loss_function(logits, batch["target"].to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(
            f"Encoder {epoch + 1}/{settings['encoder_epochs']}: loss={total / settings['encoder_steps']:.6f}",
            flush=True,
        )
    del optimizer, representation_head
    model.encoder.requires_grad_(False)
    matrix = extract_features(model.encoder, dataset, options, output / "features.npy")
    torch.manual_seed(options["seed"])
    model.readout = ContextualGatedReadout(
        model.encoder.output_dimension,
        options["architecture"]["bottleneck"],
        options["architecture"]["width"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.readout.parameters(),
        lr=settings["readout_learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    labels = np.asarray([row[2] for row in dataset.rows], dtype=np.float32)
    for epoch in range(settings["readout_epochs"]):
        model.readout.train()
        indices = sample_indices(
            dataset,
            settings["readout_steps"] * settings["readout_batch_size"],
            options["seed"] + epoch,
        )
        total = 0.0
        for ids in indices.reshape(settings["readout_steps"], settings["readout_batch_size"]):
            features = torch.from_numpy(np.array(matrix[ids], copy=True)).to(device)
            target = torch.from_numpy(labels[ids]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model.readout(features), target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite readout loss")
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(
            f"Readout {epoch + 1}/{settings['readout_epochs']}: loss={total / settings['readout_steps']:.6f}",
            flush=True,
        )
    torch.save(
        {
            "architecture": options["architecture"],
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        },
        output / "cytocue.pt",
    )
