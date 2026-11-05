# Python training

Open `train.py` here in Spyder and use F5, or run `python -m training.train` from the repository root. Open `train_final.py` for the single final training/export. The repository root, not this directory, contains local datasets and outputs.

| Module | Responsibility |
|---|---|
| config.py | Dataclasses, repository-root paths and serialization |
| data.py | WAV indexing, integrity, labels, grouped validation |
| features.py | Official-source MFCC front-end and feature cache |
| models.py | Configurable CNN1D/CNN2D construction |
| experiments.py | Training, checkpoint selection and saved-run evaluation |
| cross_validation.py | Grouped folds and experiment scheduling |
| metrics.py / keras_metrics.py | Classification metrics and training EVA AUC |
| cv_reports.py | Fold/aggregate histories and confusion matrices |
| campaigns.py | Matched three-class plan and binary comparisons |
| third_party/ | Python DSP implementation with original notices |

Data/cache/results paths derive from `Path(__file__).resolve().parents[1]`. Existing artifacts at the repository root remain usable. The reorganization does not change the MFCC math, architecture, dataset, or checkpoint selection. The typo `MODE='sinlge'` has been corrected to `single`; current user configuration remains binary with the compact 13/40 CNN1D 8/16 k5 candidate.

Main grid controls: FEATURES, GRID_FAMILIES, GRID_FILTERS, GRID_KERNELS and GRID_LEARNING_RATES. SELECTED_CONFIGURATIONS specifies exact feature/model pairs. The three-class campaign overrides training settings per case. Validation selects weights; repeated outer-fold comparisons are development/model-selection estimates.
