import hashlib
import json
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from .imaging import TiffReader, mask_cells


def crop_bounds(row, multiplier):
    _, _, _, y0, x0, y1, x1, _ = row
    side = math.ceil(max(16, 1.25 * max(y1 - y0, x1 - x0))) * multiplier
    top = math.floor((y0 + y1) / 2 - side / 2)
    left = math.floor((x0 + x1) / 2 - side / 2)
    return top, left, top + side, left + side


def read_labels(pair, root):
    spec = pair.get("labels")
    if spec is None:
        return None
    frame = pd.read_csv(root / spec["path"])
    for column, value in spec.get("filters", {}).items():
        frame = frame.loc[frame[column].astype(str) == str(value)]
    ids = pd.to_numeric(frame[spec["id"]], errors="raise")
    values = frame[spec["target"]]
    if spec.get("encoding") == "pos_neg":
        values = values.astype(str).str.lower().map({"pos": 1, "neg": 0})
    if spec.get("ambiguous"):
        valid = values.isin([0, 1])
        if not values.isin([0, 1, 2, 3]).all():
            raise ValueError("Unknown expression label")
        ids, values = ids[valid], values[valid]
    if not values.isin([0, 1]).all() or not np.equal(ids, np.floor(ids)).all() or (ids <= 0).any():
        raise ValueError("Labels must use positive integer cell IDs and binary targets")
    labels = pd.DataFrame({"cell_id": ids.astype(np.int64), "target": values.astype(np.int64)})
    if labels.groupby("cell_id").target.nunique().gt(1).any():
        raise ValueError("Conflicting cell annotations")
    return labels.drop_duplicates().sort_values("cell_id")


def prepare_pair(pair, root, cache):
    paths = [root / pair[name]["path"] for name in ("image", "mask")]
    if pair.get("labels"):
        paths.append(root / pair["labels"]["path"])
    signature = json.dumps(
        [pair, [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]],
        sort_keys=True,
    )
    destination = cache / (hashlib.sha256(signature.encode()).hexdigest() + ".npy")
    if destination.exists():
        return np.load(destination, allow_pickle=False)
    image, mask = TiffReader(paths[0]), TiffReader(paths[1])
    try:
        if image.shape != mask.shape or list(image.shape) != pair["image"]["shape"]:
            raise ValueError("Image and instance-mask dimensions do not match the manifest")
        geometry_path = cache / (
            hashlib.sha256(
                str(
                    (paths[1].resolve(), paths[1].stat().st_size, paths[1].stat().st_mtime_ns)
                ).encode()
            ).hexdigest()
            + ".npz"
        )
        if geometry_path.exists():
            with np.load(geometry_path) as saved:
                geometry = {int(row[0]): row[1:].tolist() for row in saved["cells"]}
        else:
            geometry = mask_cells(mask)
            np.savez_compressed(
                geometry_path,
                cells=np.asarray([[k, *v] for k, v in geometry.items()], dtype=np.int64).reshape(
                    -1, 6
                ),
            )
        labels = read_labels(pair, root)
        if labels is None:
            labels = pd.DataFrame({"cell_id": sorted(geometry), "target": -1})
        if pair.get("n_labels") is not None and len(labels) != pair["n_labels"]:
            raise ValueError("Annotation count differs from the manifest: " + pair["unit"])
        if not set(labels.cell_id).issubset(geometry):
            raise ValueError("Annotated cell IDs are missing from the instance mask")
        rows = np.asarray(
            [[r.cell_id, r.target, *geometry[r.cell_id]] for r in labels.itertuples()],
            dtype=np.int64,
        ).reshape(-1, 7)
        np.save(destination, rows, allow_pickle=False)
        return rows
    finally:
        image.close()
        mask.close()


class CellDataset(Dataset):
    def __init__(
        self, manifest, root, cache, split="evaluation", marker=None, cell_size=96, context_size=224
    ):
        self.root = Path(root)
        self.cell_size, self.context_size = cell_size, context_size
        cache = Path(cache)
        cache.mkdir(parents=True, exist_ok=True)
        records = json.loads(Path(manifest).read_text(encoding="utf8"))
        self.pairs = {
            p["pair_key"]: p
            for p in records
            if p["split"] == split and (marker is None or p["marker"] == marker)
        }
        if not self.pairs:
            raise ValueError("No image-marker pairs match the requested split and marker")
        self.rows = []
        for key, pair in self.pairs.items():
            self.rows.extend((key, *map(int, row)) for row in prepare_pair(pair, self.root, cache))
        self.readers = OrderedDict()

    def __len__(self):
        return len(self.rows)

    def _reader(self, key, which):
        path = self.root / self.pairs[key][which]["path"]
        if path not in self.readers:
            self.readers[path] = TiffReader(path)
        self.readers.move_to_end(path)
        while len(self.readers) > 8:
            self.readers.popitem(last=False)[1].close()
        return self.readers[path]

    def close(self):
        for reader in self.readers.values():
            reader.close()
        self.readers.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["readers"] = OrderedDict()
        return state
