# Workflow

## Prepared datasets: recommended starting point

The original Colab analyses read two event-level Parquet files with EMS-RRI scores, contextual predictors, and coded outcome fields. Use those original files when reproducing the paper; do not regenerate or reorder them unnecessarily. `tools/check_inputs.py` checks core required columns, sample sizes, and training-derived cutoffs without fitting a model. It is not an exhaustive schema validator for every exploratory audit.

Each script/notebook is independently configured. Dependent downstream bootstrap cells and the Frontier extension are kept with their parent analyses. Run the numbered scripts individually; scripts with later numbers generally refit their own models rather than consuming earlier fitted models.

| Analysis | Purpose | Manuscript output |
|---|---|---|
| 01 | Final 32-predictor forward model | Table 6 point estimates |
| 02 | Percentile-cutoff sensitivity | Table 4 |
| 03 | Failure distances, high-recall threshold, three tiers | Tables 11–12 |
| 04 | Bootstrap uncertainty, calibration, decision analysis | Tables 6–8; Figure 3 |
| 05 | Refitted predictor-domain ablation | Figure 4 source results |
| 06 | Forward high-acuity comparison, paired CIs, selective gating | Tables 13–14 |
| 07 | Subgroups, drift, Frontier exploratory analysis | Table 10; Figure 5 source results |
| 08 | Scoring-element/proxy audit and documentation-state ablation | Table 9 |
| 09 | Validation-only 34 vs. 32 predictor selection | Table 5 XGBoost comparison |
| 10 | Score distributions and agreement | Table 3 |
| 11 | SGD baseline | Table 5 SGD comparison |
| 12 | Forward dead-without-resuscitation sensitivity | Table 15 |
| 13 | Class-weighting sensitivity | Downstream sensitivity text |

Some cells print results rather than saving tables. This release does not contain separate source files for every publication-layout figure or conceptual diagram. The mapping identifies underlying analyses, not a guarantee of identical figure formatting.

## Raw-data preparation

Use a private work directory and the original NEMSIS releases. The sampling/filtering scripts require the `unzip` command. They stream large archives and are not small demonstrations.

1. Set `NEMSIS_WORK_DIR` to the private root containing the raw ZIPs. The supplied scripts expect `ASCII_64 2024.zip` and `ASCII 25.zip`; adjust these filename settings for your authorized copies.
2. `preprocessing/01_sample_events_2024.py` samples 200,000 2024 events. The original filename misleadingly said 500k; its active settings are 2024/200k. For 2025, the original source only contains commented alternative settings. To regenerate 2025, explicitly set its ZIP, sample size 500,000, `samples_2025` output directory, and filenames `Pub_PCRevents_500k_random.txt` / `pcrkeys_500k_random.txt`. Preserve the sampling algorithm, raw-file order, and seed 42. An independent finalized 2025 sampler was not supplied.
3. Run `02_filter_related_2025.py` and `03_filter_related_2024.py` after their corresponding event/key samples exist. Their filenames identify years, not dependency order.
4. Run `04_build_clean_merged_datasets.py` to create the two files under `simple_clean_merged/`.
5. Run `05_calculate_ems_rri.py` to create scores under `final_ems_rri_datasets/`, plus rule and calculation audits. Scores are rounded to four decimal places by the supplied scorer.
6. The exact final export/rename step to the Colab `datasets/nemsis_2024.parquet` and `nemsis_2025.parquet` was not supplied. Use the existing original Colab inputs, or verify any reconstructed export against them before claiming replication. Do not substitute the archived experiment builder's split/label columns: it uses the superseded year direction.

Optional field inventories, audits, and original preprocessing sources remain in `archive/scripts*`. Some require NEMSIS dictionaries or intermediate inventory files not included in this public code package. They are not required by the final scorer, whose rules are embedded explicitly.

## Colab

Upload/unzip this repository into your Colab environment or clone your eventual repository. Mount Google Drive yourself if the private datasets are there. In a setup cell, set the two paths before running an organized notebook:

```python
import os
os.environ['EMS_RRI_REPOSITORY_ROOT'] = '/content/emsrri-github'
os.environ['EMS_RRI_ANALYSIS_ROOT'] = '/content/drive/MyDrive/NEMSIS_NEW'
```

Install `requirements.txt` in that environment. Automatic Drive mounting was removed from organized notebooks so the same code can run locally. Historical snapshots still preserve their original Colab setup cells.
