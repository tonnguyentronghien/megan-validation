# BDI Framework — Reproducibility Package

Code, data, and results for the paper:

> **Evaluating the Incorporation of BWM's Importance Weights into DEMATEL-ISM**

This repository contains everything needed to reproduce every table and figure
in the manuscript, plus the Round-2 supplementary analyses (statistical
testing, weight-validity analysis, scalability study, and the
weighting-method generalisation experiments).

## Repository structure

```
├── data/
│   ├── primary_direct_relation_matrix_9x9.csv    # Primary dataset (9 barriers, Ojha et al.)
│   ├── primary_bwm_weights.csv                   # BWM weight vector (Table I setting)
│   └── secondary_chen2021_matrix_15x15.csv       # Cross-domain dataset (Chen 2021, 15 factors)
├── notebooks/                                    # Original analysis notebooks (as run for the manuscript)
│   ├── Paper3.ipynb                              # BDI-M pipeline: weighting, DEMATEL, MMDE/MEAN+SD, ISM, Monte Carlo, Fig. 2/3 code
│   ├── Paper3_2.ipynb                            # BDI-F pipeline (8x8), thresholds, ISM tables; DI baseline
│   └── Paper3_3.ipynb                            # Cross-domain validation (Table IX), Table VIII diagnostics, verification cell
├── scripts/
│   ├── 01_full_verification.py                   # Single-file check of every published value
│   │                                             #   (Tables IV, V, VII, VIII, IX; Fig. 2 statistics;
│   │                                             #    evidence for the corrected DI-MMDE value 0.179)
│   ├── 02_R4_supplementary_analysis.py           # Round-2 supplementary analyses (Sections S0-S5)
│   └── 03_regenerate_manuscript_figs.py          # Regenerates manuscript Fig. 2 and Fig. 3
├── results/                                      # CSV outputs of script 02
├── figures/                                      # All five computational figures:
│   #  fig2_threshold_sensitivity.png   – manuscript Fig. 2 (script 03)
│   #  fig3_stability_index.png         – manuscript Fig. 3 (script 03)
│   #  fig_R4_S2_stability_tests.png    – manuscript Fig. 4 (script 02)
│   #  fig_R4_S5_dose_response.png      – manuscript Fig. 5 (script 02)
│   #  fig_R4_S3_scalability.png        – manuscript Fig. 6 (script 02)
│   #  (manuscript Fig. 1 is a hand-drawn conceptual diagram, not code-generated)
├── run_output_verification.txt                   # Console log of script 01 (all checks PASS)
├── run_output_R4.txt                             # Console log of script 02
└── requirements.txt
```

## How to run

```bash
pip install -r requirements.txt
python scripts/01_full_verification.py          # ~2 min; prints PASS/FAIL for every published value
python scripts/02_R4_supplementary_analysis.py  # ~1-2 min; writes results/*.csv and figures (Figs. 4-6)
python scripts/03_regenerate_manuscript_figs.py # ~1 min; regenerates manuscript Figs. 2-3
```

All stochastic procedures use **fixed seeds**, so both scripts are bit-for-bit
reproducible. Tested with Python 3.11+ (numpy 2.x, scipy 1.17, pandas 3.0,
matplotlib 3.10); the code uses no version-specific features and also runs on
the numpy 1.x / pandas 2.x stack shipped with Anaconda.

## What each script reproduces

**`01_full_verification.py`** — reproduction of the manuscript:

| Section | Reproduces |
|---|---|
| 1a | Prominence (D+R) and Relation (D-R), BDI-M — Table IV |
| 1b | MEAN+SD and MMDE thresholds for DI / BDI-M / BDI-F — Table V |
| 1c | MMDE candidate scan: evidence that 0.504 was a transcription error; corrected optimum 0.179 |
| 1d | ISM hierarchy levels and variance, six configurations — Table VII |
| 2  | Cross-domain validation, Chen 2021 (surrogate BWM, thresholds, ISM, reclassification) — Table IX |
| 3  | Monte Carlo convergence diagnostics — Table VIII |
| 4  | MEAN+SD threshold distribution statistics — Fig. 2 |

**`02_R4_supplementary_analysis.py`** — Round-2 supplementary analyses:

| Section | Content | Reviewer comment |
|---|---|---|
| S0 | Reproduction check of all published threshold values | — |
| S1 | Validity of the BWM weight vector: reverse elicitation + consistency ratio (CR_I = 0.0057), convergent/discriminant validity, 10,000 synthetic expert panels | Comment 2 |
| S2 | Paired statistical tests (exact McNemar, Wilcoxon signed-rank, 10,000-resample bootstrap CIs, rank-biserial effect sizes, Holm correction) under two Monte Carlo formulations | Comment 3 |
| S3 | Scalability n = 9…200, empirical complexity exponents, O(m log m) MMDE re-implementation (equivalence-checked) | Comment 4 |
| S4 | Quantitative decision diagnostics and the two-step BDI-M/BDI-F + threshold selection guideline | Comment 5 |
| S5 | Generalisation of the MMDE degeneration to AHP, ANP (DANP), and FUCOM weights; homogeneous-vs-heterogeneous scaling control; dose-response curve | Comment 7 |

## Data provenance

* **Primary dataset** (9 barriers): mean direct-relation matrix adapted from
  Ojha et al.; BWM weight vector as specified in the manuscript.
* **Secondary dataset** (15 factors): direct-relation matrix published by
  Chen (2021), social-insurance participation domain; used for the
  cross-domain validation in Section III.D of the manuscript.

## Citation

If you use this code or data, please cite the paper above. A DOI-stamped
archive of this repository is deposited on Zenodo.
