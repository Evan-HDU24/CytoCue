import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from cytocue.data import CellDataset
from cytocue.inference import predict
from cytocue.metrics import summarize
from cytocue.model import load_model


def main():
    parser = argparse.ArgumentParser(description="CytoCue cell-level marker classification")
    parser.add_argument("command", choices=["prepare", "predict", "evaluate", "train"])
    parser.add_argument("--options", default="options.json")
    parser.add_argument("--data-root")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--device")
    parser.add_argument("--split", choices=["train", "evaluation"])
    args = parser.parse_args()
    path = Path(args.options).resolve()
    options = json.loads(path.read_text(encoding="utf8"))
    for key in ("data_root", "checkpoint", "output", "device"):
        value = getattr(args, key)
        if value is not None:
            options[key] = value
    for key in ("data_root", "checkpoint", "output", "dataset_manifest"):
        value = Path(options[key])
        options[key] = str(value if value.is_absolute() else path.parent / value)
    if options["precision"] not in ("bf16", "float32"):
        raise ValueError("precision must be bf16 or float32")
    device = torch.device(options["device"])
    if args.command != "prepare" and device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu for inference")
    torch.set_num_threads(1)
    random.seed(options["seed"])
    np.random.seed(options["seed"])
    torch.manual_seed(options["seed"])
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    output = Path(options["output"])
    output.mkdir(parents=True, exist_ok=True)
    split = args.split or ("train" if args.command == "train" else "evaluation")
    if args.command == "train" and split != "train":
        raise ValueError("Training accepts only the training split")
    dataset = CellDataset(
        options["dataset_manifest"],
        options["data_root"],
        output / "index",
        split,
        None if split == "train" else options["marker"],
        options["cell_size"],
        options["context_size"],
    )
    if args.command == "prepare":
        print(f"Prepared {len(dataset)} cells across {len(dataset.pairs)} image-marker pairs")
    elif args.command == "train":
        from cytocue.training import train

        train(dataset, options)
    else:
        model = load_model(options["checkpoint"], device)
        predictions = predict(model, dataset, options)
        predictions.to_csv(output / "predictions.csv", index=False)
        if args.command == "evaluate":
            images, blocks, scores = summarize(predictions, options["threshold"])
            images.to_csv(output / "metrics_by_image.csv", index=False)
            blocks.to_csv(output / "metrics_by_block.csv")
            (output / "metrics.json").write_text(json.dumps(scores, indent=2), encoding="utf8")
            print(json.dumps({k: round(v * 100, 4) for k, v in scores.items()}, indent=2))
    dataset.close()


if __name__ == "__main__":
    main()
