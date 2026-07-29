# BDI Framework — Reproducibility Package

Code, data, and results for the paper:

> **Evaluating the Incorporation of BWM's Importance Weights into DEMATEL-ISM**

This repository contains everything needed to reproduce every computational
table and figure in the manuscript (Tables IV–XI; Figs. 2–6), plus the
supplementary analyses added in the second revision: statistical testing,
weight-validity analysis, scalability study, and the weighting-method
generalisation experiments. All stochastic procedures use **fixed random
seeds**, so every result is bit-for-bit reproducible.

## Repository contents

**Scripts** (run from the repository root)

| File | Purpose |
|---|---|
| `01_full_verification.py` | Single-file check of every published value: Tables IV, V, VII, VIII, IX; Fig. 2 statistics; evidence for the corrected DI-MMDE threshold (0.179). Prints PASS/FAIL for each check. |
| `02_R4_supplementary_analysis.py` | Revision-2 supplementary analyses (Sections III.C, III.E–III.G of the manuscript): statistical tests (Table X, Fig. 4), weight-validity analysis, scalability study (Fig. 6), and the AHP/ANP/FUCOM generalisation (Table XI, Fig. 5). Writes CSV outputs to `results/` and figures to `figures/`. |
| `03_regenerate_manuscript_figs.py` | Regenerates manuscript Fig. 2 and Fig. 3 with the original notebook code paths. |

**Original analysis notebooks** (as run during the study; kept for provenance)

| File | Content |
|---|---|
| `Paper3_bwm.ipynb` | BDI-M pipeline on the primary dataset: BWM weight embedding, DEMATEL, MMDE vs MEAN+SD thresholds, ISM partition, Monte Carlo simulation. |
| `Paper3_2.ipynb` | BDI-F pipeline (8×8 reduced model): total relation matrix, thresholds, ISM tables; DI baseline. |
| `Paper3_3.ipynb` | Cross-domain validation on the secondary dataset (Table IX), Monte Carlo convergence diagnostics (Table VIII), and the full verification cell. |

**Data**

| File | Description |
|---|---|
| `primary_direct_relation_matrix_9x9.csv` | Primary dataset: mean direct-relation matrix, 9 barriers (adapted from Ojha et al.). |
| `primary_bwm_weights.csv` | BWM weight vector of the primary analysis. |
| `secondary_chen2021_matrix_15x15.csv` | Secondary dataset: direct-relation matrix, 15 factors (Chen, 2021). |

**Figures** — the exact files used in the manuscript

| File | Manuscript figure |
|---|---|
| `fig1_framework.png` | Fig. 1 — conceptual framework (diagram, not code-generated) |
| `fig2_threshold_sensitivity.png` | Fig. 2 — MEAN+SD threshold distributions under perturbation |
| `fig3_stability_index.png` | Fig. 3 — Stability Index: BDI-M vs BDI-F vs DI |
| `fig_R4_S2_stability_tests.png` | Fig. 4 — SI with bootstrap 95% CIs (paired tests) |
| `fig_R4_S5_dose_response.png` | Fig. 5 — dose-response of the MMDE degeneration |
| `fig_R4_S3_scalability.png` | Fig. 6 — runtime scalability, n = 9–200 |

**Logs** — `run_output_verification.txt` (script 01, all checks PASS) and
`run_output_R4.txt` (script 02, full console output) document a complete
reference run.

## How to run

```bash
pip install -r requirements.txt
python 01_full_verification.py           # ~2 min; prints PASS/FAIL for every published value
python 02_R4_supplementary_analysis.py   # ~1–2 min; writes results/*.csv and figures/*.png (Figs. 4–6)
python 03_regenerate_manuscript_figs.py  # ~1 min; regenerates Figs. 2–3 into figures/
```

Requirements: Python 3.9+ with numpy, scipy, pandas, matplotlib (any recent
versions; tested on numpy 1.x/2.x). The scripts create the `results/` and
`figures/` output folders automatically; the numerical outputs they produce
are identical on every run and on every machine, except for the runtimes in
the scalability section, which are hardware-dependent (the log–log
complexity exponents are the machine-invariant quantity).

## Mapping to the Round-2 reviewer comments

| Reviewer comment | Where it is addressed |
|---|---|
| Comparison with prior BWM–DEMATEL studies | Manuscript Introduction + Table I |
| Validity of the assumed BWM weights | Script 02, Section S1 → manuscript Section III.F |
| Statistical testing (McNemar, Wilcoxon, bootstrap) | Script 02, Section S2 → manuscript Section III.C, Table X, Fig. 4 |
| Computational efficiency at large n | Script 02, Section S3 → manuscript Section III.G, Fig. 6 |
| BDI-M vs BDI-F selection guidance | Script 02, Section S4 → manuscript Section III.E (two-step guideline) |
| Public repository | This repository (MIT license, fixed seeds) |
| Mechanism; AHP/ANP/FUCOM generalisation | Script 02, Section S5 → manuscript Section III.E, Table XI, Fig. 5 |

## Data provenance

The primary dataset (9 barriers) is the mean direct-relation matrix adapted
from Ojha et al.; the BWM weight vector is specified in the manuscript. The
secondary dataset (15 factors) is the direct-relation matrix published by
Chen (2021) for social-insurance participation, used for the cross-domain
validation in Section III.D.

## License and citation

Released under the MIT License. If you use this code or data, please cite
the paper above. A version-stamped archive of this repository is deposited
on Zenodo (DOI added upon acceptance).
