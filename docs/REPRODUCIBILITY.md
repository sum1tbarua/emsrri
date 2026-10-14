# Reproducibility boundaries

## Preserved scientific implementation

The organized analyses retain the supplied model configurations, feature lists, seeds, split calls, thresholds, bootstrap logic, and metric calculations. Packaging changes concern paths, notebook organization, and removing automatic Colab mounting. No reported result has been recomputed or silently corrected.

The supplied scorer rounds both scores to four decimal places. Preserve that behavior: rounding and equality at discrete cutoffs can change labels. The weighting uses inverse selected-domain size; applicability still changes each record's realized contribution. Missing triggers do not activate conditional fields. The raw-NOT_APPLICABLE flag applies to default-rule fields; conditional rules use the trigger test alone. These are implementation details, not independent clinical validation of the rules.

## Gaps that remain

- Original dataset files, sampled event keys, final input hashes, and frozen software versions were not supplied. Exact sample/split and numerical replication cannot be certified from source alone.
- The final handoff to Colab input filenames is undocumented. The archived experiment builder uses a superseded reverse-temporal design. It must not define the final splits or cutoffs.
- Analysis 08 originally required `Dataset_Investigation.ipynb`, which was not supplied. Its source-element enumeration now reads a clearly labeled reference notebook containing the exact `RULES` assignment from the supplied final scorer. The original enumeration and domain-count checks are retained. This replaces the missing reference dependency, not the model or ablation procedure.
- The omnibus source notebook includes superseded downstream analyses and diagnostic cells. Those remain only in the historical snapshot. The organized release uses its forward downstream cells, including the final forward dead-resuscitation analysis.
- A seed alone does not fix splits if input row order changes. New dependency versions can change fitted predictions. Do not describe the unpinned requirements as the original environment.
- Analyses 10 and 11 retain their print-oriented output behavior. Publication figure styling and every table export are not automatically reproduced by this package.
- The source archives include exploratory logic, old path settings, and external-file dependencies. Their preservation is for provenance, not a promise that all archived files run independently.

## Interpretation

EMS-RRI is a selected-field completeness measure. Its score-derived labels, model discrimination, tier boundaries, and review simulations do not establish correctness of records or clinical benefit. The primary downstream selective gate was negative. All operational applications require separate evaluation.

## Before claiming a reproduced release

Record hashes of the private original input files, retain their row order, capture `python -m pip freeze`, rerun the intended analyses, and compare their outputs with the paper. Keep record-level predictions and identifiers private. Add the final paper citation and an author-approved code license. None of these missing details has been invented in this package.
