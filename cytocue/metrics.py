import numpy as np
import pandas as pd


def summarize(predictions, threshold=0.5):
    rows = []
    for (block, image), group in predictions.groupby(["block", "image"]):
        y = group.target.to_numpy()
        if not np.isin(y, [0, 1]).all():
            raise ValueError("Evaluation requires binary annotations for every cell")
        p = group.probability.to_numpy() >= threshold
        tp = int(((y == 1) & p).sum())
        fp = int(((y == 0) & p).sum())
        fn = int(((y == 1) & ~p).sum())
        tn = int(((y == 0) & ~p).sum())
        rows.append(
            {
                "block": block,
                "image": image,
                "cells": len(y),
                "accuracy": (tp + tn) / len(y),
                "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan,
                "precision": tp / (tp + fp) if tp + fp else np.nan,
                "recall": tp / (tp + fn) if tp + fn else np.nan,
            }
        )
    images = pd.DataFrame(rows)
    columns = ["accuracy", "f1", "precision", "recall"]
    blocks = images.groupby("block")[columns].mean()
    return images, blocks, blocks.mean().to_dict()
