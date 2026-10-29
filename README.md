# CytoCue

Cell-level marker positivity classification from multiplexed tissue images and aligned cell-instance masks. CytoCue combines Cell–Neighborhood Encoding (CNE) with a Contextual Gated Readout (CGR).

## Run

The verified environment uses Python 3.12 and PyTorch 2.8 / torchvision 0.23 with CUDA 12.8. Install a compatible PyTorch build, then install the remaining dependencies:

```bash
pip install -r requirements.txt
python main.py prepare --data-root /path/to/data
python main.py evaluate --data-root /path/to/data
```

Run commands through `main.py` and adjust paths and parameters in [options.json](options.json). Evaluation saves predictions and ACC, F1, Precision and Recall to `outputs/`.

## Data

Images, masks and cell annotations are not distributed here. Obtain the original releases and cite their authors:

- [Pan-Multiplex-Gold-Standard](https://huggingface.co/datasets/JLrumberger/Pan-Multiplex-Gold-Standard/tree/0cad0b3231c190695ae8424fdef57d74ee0e7026): Rumberger et al., *Automated classification of cellular expression in multiplexed imaging data with Nimbus*, Nature Methods (2025), [doi:10.1038/s41592-025-02826-9](https://doi.org/10.1038/s41592-025-02826-9).
- [PhenoBIC-Data](https://huggingface.co/datasets/mittal-research-lab/PhenoBIC-Data/tree/aeffd288d922ee5df4cbb414f3c1f77507ba0001): Sankaranarayanan et al., *PhenoBIC: operator-free single-cell spatial phenotyping in multiplex imaging data using deep learning of cell staining patterns* (2026).

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

Keep the original folder structure when extracting the archives. [datasets.json](datasets.json) specifies the files used by the project. For your own data, provide aligned TIFF images and cell-instance masks and update this file accordingly.

Training also requires the Pan-Multiplex MIBI decidua, Vectra colon and Vectra pancreas data, and the PhenoBIC training images, masks and annotations from the same sources.

## Training

```bash
python main.py train --data-root /path/to/data --output outputs/training
```

## References and citation

Please cite the original data sources when using their datasets. Related references are provided in [references.bib](references.bib), and third-party acknowledgements are listed in [NOTICE](NOTICE).

To cite this code repository:

```bibtex
@misc{cytocue2026,
  title = {{CytoCue: Cell-Level Marker Positivity Classification}},
  year = {2026},
  howpublished = {GitHub repository},
  url = {https://github.com/Evan-HDU24/CytoCue}
}
```
