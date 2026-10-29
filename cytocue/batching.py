from .data import crop_bounds
from collections import defaultdict
import numpy as np
import torch
from torch.nn import functional as F

MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).reshape(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).reshape(1, 3, 1, 1)


def context_batch(base, indices, include_valid=False):
    n = len(indices)
    context_size = getattr(base, "context_size", 96)
    out = {
        "index": torch.as_tensor(indices, dtype=torch.int64),
        "target": torch.tensor([base.rows[i][2] for i in indices], dtype=torch.float32),
    }
    mask_names = ["target", "neighbor"] + (["valid"] if include_valid else [])
    for view, size in [("fine", base.cell_size), ("context", context_size)]:
        out[view + "_gray"] = torch.empty((n, 1, size, size), dtype=torch.float32)
        for name in mask_names:
            out[view + "_" + name] = torch.empty((n, 1, size, size), dtype=torch.uint8)
    groups = defaultdict(list)
    for destination, i in enumerate(indices):
        row = base.rows[i]
        key, cid = row[:2]
        outer = crop_bounds(row, 2)
        im = base._reader(key, "image")
        seg = base._reader(key, "mask")
        image = im.read(*outer).astype("float32", copy=False)
        ids = seg.read(*outer)
        if not np.isfinite(image).all():
            raise ValueError("Nonfinite input image")
        valid = im.coverage(*outer) & seg.coverage(*outer)
        q = base.pairs[key]["normalization"]
        normalized = np.clip((image - q["q1"]) / max(q["q99_9"] - q["q1"], 1e-06), 0, 1)
        target = (ids == cid) & valid
        neighbor = (ids > 0) & ~target & valid
        masks = [target, neighbor] + ([valid] if include_valid else [])
        for view, mult, size in [("fine", 1, base.cell_size), ("context", 2, context_size)]:
            y0, x0, y1, x1 = crop_bounds(row, mult)
            a, b = (y0 - outer[0], x0 - outer[1])
            c, d = (y1 - outer[0], x1 - outer[1])
            gray = normalized[a:c, b:d]
            ms = np.stack([v[a:c, b:d] for v in masks]).astype(np.uint8)
            groups[view, size, gray.shape].append((destination, gray, ms))
    for (view, size, shape), items in groups.items():
        destinations = torch.tensor([x[0] for x in items], dtype=torch.int64)
        gray = torch.from_numpy(np.stack([x[1] for x in items]))[:, None]
        image = F.interpolate(gray, size=(size, size), mode="bilinear", align_corners=False)
        masks = F.interpolate(
            torch.from_numpy(np.stack([x[2] for x in items])), size=(size, size), mode="nearest"
        )
        out[view + "_gray"].index_copy_(0, destinations, image)
        for c, name in enumerate(mask_names):
            out[view + "_" + name].index_copy_(0, destinations, masks[:, c : c + 1])
    return out


def rgb_from_gray(gray, device):
    gray = gray.to(device, non_blocking=True)
    with torch.autocast(
        device_type="cuda" if str(device).startswith("cuda") else "cpu", enabled=False
    ):
        image = (gray.repeat(1, 3, 1, 1) - MEAN.to(device)) / STD.to(device)
    return image.contiguous(memory_format=torch.channels_last)


def move_context(batch, device="cuda"):
    out = {view: rgb_from_gray(batch[view + "_gray"], device) for view in ["fine", "context"]}
    for key, value in batch.items():
        if key.startswith(("fine_", "context_")) and (not key.endswith("_gray")):
            out[key] = value.to(device, non_blocking=True).float()
    return out
