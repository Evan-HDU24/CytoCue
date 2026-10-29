# CytoCue

Cell-level marker positivity classification from multiplexed tissue images and aligned cell-instance masks. CytoCue combines Cell–Neighborhood Encoding (CNE) with a Contextual Gated Readout (CGR).

## Run

The verified environment uses Python 3.12 and PyTorch 2.8 / torchvision 0.23 with CUDA 12.8. Install a compatible PyTorch build, then install the remaining dependencies:

```bash
pip install -r requirements.txt
python main.py prepare --data-root /path/to/data
python main.py evaluate --data-root /path/to/data
```

`main.py` is the only entry point. `options.json` contains paths, input sizes, feature dimensions, batch sizes, the random seed and training settings. Evaluation writes cell probabilities and ACC, F1, Precision and Recall to `outputs/`. Use `predict` for probability export without scoring, or `--device cpu` for CPU inference. CUDA with BF16 and batch size 256 matches the released inference configuration.

## Model

```text
marker image + instance mask
          ↓
cell view (96×96) + context view (224×224)
          ↓
CNE: local target / contextual target / contextual neighbours
          ↓
CGR: feature compression and multiplicative gating
          ↓
cell positivity probability
```

`cytocue/encoding.py` implements CNE, `readout.py` implements CGR, and `model.py` combines them. `data.py`, `imaging.py` and `batching.py` handle cell geometry and aligned image crops. `inference.py`, `metrics.py` and `training.py` provide prediction, scoring and fixed-epoch training.

The single checkpoint, `weights/cytocue.pt`, contains the complete model; inference does not download backbone weights. It corresponds to the reported CytoCue configuration: ACC 99.29%, F1 91.79%, Precision 89.54% and Recall 94.75%. Scores use threshold 0.5, image-level averaging within each block, then an equal mean over the four CD8 blocks (48 images, 456,808 labelled cells).

The released weights combine encoder epoch 15 and readout epoch 2, selected during model development. These are development-benchmark results, not an untouched holdout estimate. The training command below runs the fixed schedule and saves its final parameters; it does not reproduce the historical checkpoint-selection procedure or guarantee the same score.

## Data

Images, masks and cell annotations are not distributed here. Obtain the original releases and cite their authors:

- [Pan-Multiplex-Gold-Standard](https://huggingface.co/datasets/JLrumberger/Pan-Multiplex-Gold-Standard/tree/0cad0b3231c190695ae8424fdef57d74ee0e7026): Rumberger et al., *Automated classification of cellular expression in multiplexed imaging data with Nimbus*, Nature Methods (2025), [doi:10.1038/s41592-025-02826-9](https://doi.org/10.1038/s41592-025-02826-9).
- [PhenoBIC-Data](https://huggingface.co/datasets/mittal-research-lab/PhenoBIC-Data/tree/aeffd288d922ee5df4cbb414f3c1f77507ba0001): Sankaranarayanan et al., *PhenoBIC: operator-free single-cell spatial phenotyping in multiplex imaging data using deep learning of cell staining patterns* (2026). See `references.bib` and the original repository for attribution and data terms.

Extract `gold_standard_labelled.zip` and `external_val_TMA_cores.zip`; obtain the three `TMA*_CD8.csv` annotation files. Arrange the evaluation files as follows:

```text
data/
  panm/gold_standard_labelled/
    gold_standard_groundtruth.csv
    codex_colon/{fovs,masks}/
    mibi_breast/{fovs,masks}/
  phenobic/
    external_val_TMA_cores/
    ground_truth/external_val_TMA_cores/TMA1_CD8.csv
    ground_truth/external_val_TMA_cores/TMA2_CD8.csv
    ground_truth/external_val_TMA_cores/TMA3_CD8.csv
```

Preserve the internal archive paths. `datasets.json` lists the exact image-marker pairs, annotation filters, counts and normalization constants; it contains no images, masks or cell labels. Ambiguous Pan-Multiplex annotations are excluded. TMA1 and TMA2 form one block. A missing source is reported as an error rather than silently omitted. The Pan-Multiplex dataset card has an availability note for MIBI breast; ensure the pinned release actually contains these files before attempting the full benchmark.

For other images, supply a manifest using the same schema. Each image and mask must be a scalar 2-D TIFF on the same pixel grid; cell IDs are positive integers and background is zero. Set `labels` to `null` for unlabelled prediction. Supply `q1` and `q99_9` from the image intensities inside all segmented cells, not from individual crops. No display gamma or pseudocolor is used as model input.

## Training

```bash
python main.py train --data-root /path/to/data --output outputs/training
```

The default schedule is 30 encoder epochs (the first 3 freeze the backbones), followed by 10 CGR epochs with the encoder fixed. Each encoder epoch samples 1,000 batches of 64 cells; each CGR epoch samples 500 batches of 128 cells. Source and marker groups are sampled uniformly. Training uses ImageNet initialization, synchronized spatial augmentation for representation learning, and BCE plus soft Dice loss. Only the final model is saved as `outputs/training/cytocue.pt`; no evaluation-driven stopping or parameter search is included.

Training additionally requires the Pan-Multiplex MIBI decidua, Vectra colon and Vectra pancreas folders, plus PhenoBIC `multiplex_WSIs/`, `cell_segmentations/` and `ground_truth/training_WSIs/`. The manifest defines 430,536 training cell–marker observations. Feature caching uses approximately 2.5 GiB, in addition to raw images and masks. Keep the released checkpoint separate from new training outputs.

## References and weights

Bibliographic entries are in `references.bib`; dependency attribution is in `NOTICE`. The checkpoint is approximately 91 MiB. `.gitattributes` configures Git LFS for weights; install Git LFS before adding the repository to GitHub.
