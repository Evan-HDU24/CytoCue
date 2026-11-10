from collections import defaultdict

import numpy as np
import torch

from .batching import move_context
from .inference import autocast, batches


class TrainingScore:
    def __init__(self, dataset):
        groups = defaultdict(list)
        self.targets = np.asarray([row[2] for row in dataset.rows])
        for index, row in enumerate(dataset.rows):
            pair = dataset.pairs[row[0]]
            if pair["split"] != "train" or row[2] not in (0, 1):
                raise ValueError("Checkpoint selection requires annotated training cells")
            groups[(pair["resource"], pair["source"], pair["marker"])].append(index)
        if not groups:
            raise ValueError("The training set is empty")
        self.groups = [np.asarray(groups[key]) for key in sorted(groups)]

    def __call__(self, probabilities):
        probabilities = np.asarray(probabilities)
        if probabilities.shape != self.targets.shape or not np.isfinite(probabilities).all():
            raise ValueError("Selection requires finite predictions for every training cell")
        predicted = probabilities >= 0.5
        scores = []
        for ids in self.groups:
            positive = self.targets[ids] == 1
            tp = np.count_nonzero(predicted[ids] & positive)
            denominator = np.count_nonzero(predicted[ids]) + np.count_nonzero(positive)
            scores.append(2 * tp / denominator if denominator else 0.0)
        return float(np.mean(scores))


class BestTrainingState:
    def __init__(self):
        self.score = -float("inf")
        self.epoch = None
        self.state = None

    def update(self, module, score, epoch):
        if not np.isfinite(score):
            raise ValueError("Nonfinite training selection score")
        if score > self.score:
            self.score, self.epoch = float(score), int(epoch)
            self.state = {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}

    def restore(self, module):
        if self.state is None:
            raise ValueError("No completed training epoch is available")
        module.load_state_dict(self.state)


@torch.inference_mode()
def training_predictions(encoder, readout, dataset, options, matrix=None):
    device = next(readout.parameters()).device
    probabilities = np.full(len(dataset), np.nan, dtype=np.float32)
    readout.eval()
    if matrix is None:
        encoder.eval()
        for batch in batches(dataset, options["batch_size"], options["workers"], options["seed"]):
            with autocast(device, options["precision"]):
                features = encoder(move_context(batch, device)).float()
            probabilities[batch["index"].numpy()] = readout(features).reshape(-1).sigmoid().cpu().numpy()
    else:
        if len(matrix) != len(dataset):
            raise ValueError("Training features do not match the training set")
        for start in range(0, len(dataset), options["batch_size"]):
            stop = min(start + options["batch_size"], len(dataset))
            features = torch.from_numpy(np.array(matrix[start:stop], copy=True)).to(device)
            probabilities[start:stop] = readout(features).reshape(-1).sigmoid().cpu().numpy()
    return probabilities
