# 10 score agreement
# See companion notebook and docs/PROVENANCE.md.
# Repository configuration; no data are loaded by this cell.
import os
from pathlib import Path
if "EMS_RRI_REPOSITORY_ROOT" in os.environ:
    REPOSITORY_ROOT = Path(os.environ["EMS_RRI_REPOSITORY_ROOT"]).expanduser().resolve()
elif "__file__" in globals():
    REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
else:
    REPOSITORY_ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
                            if (p / "reference/scoring_rules.json").exists()), None)
    if REPOSITORY_ROOT is None:
        raise RuntimeError("Start Jupyter inside the repository, or set EMS_RRI_REPOSITORY_ROOT.")
ANALYSIS_ROOT = Path(os.environ.get("EMS_RRI_ANALYSIS_ROOT", str(REPOSITORY_ROOT / "work"))).expanduser().resolve()
ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
print("Analysis data/output root:", ANALYSIS_ROOT)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

base = str(ANALYSIS_ROOT / "datasets")
columns = ["ems_rri_basic", "ems_rri_domain_weighted"]

data = {
    2024: pd.read_parquet(f"{base}/nemsis_2024.parquet", columns=columns),
    2025: pd.read_parquet(f"{base}/nemsis_2025.parquet", columns=columns),
}

assert len(data[2024]) == 200_000
assert len(data[2025]) == 500_000

# Reproduce the 2024 training partition used in the revision.
train24, _ = train_test_split(
    data[2024], test_size=0.30, random_state=42, shuffle=True
)

cutoffs = {
    measure: train24[measure].quantile(0.20)
    for measure in columns
}

print("2024 training-derived cutoffs:")
for measure, cutoff in cutoffs.items():
    print(f"  {measure}: {cutoff:.4f}")

rows = []
for year, df in data.items():
    for measure in columns:
        scores = df[measure]
        q1, q3 = scores.quantile([0.25, 0.75])
        low_n = int(scores.le(cutoffs[measure]).sum())

        rows.append({
            "Year": year,
            "Measure": measure,
            "N": len(scores),
            "Mean": scores.mean(),
            "SD": scores.std(),
            "Q1": q1,
            "Median": scores.median(),
            "Q3": q3,
            "IQR": q3 - q1,
            "Low n": low_n,
            "Low %": 100 * low_n / len(df),
        })

    basic_low = df["ems_rri_basic"].le(cutoffs["ems_rri_basic"])
    weighted_low = df["ems_rri_domain_weighted"].le(
        cutoffs["ems_rri_domain_weighted"]
    )

    print(f"\n{year} basic–weighted comparison:")
    print(f"  Classification agreement: {100 * (basic_low == weighted_low).mean():.2f}%")
    print(f"  Cohen's kappa: {cohen_kappa_score(basic_low, weighted_low):.4f}")
    print(f"  Pearson r: {df[columns[0]].corr(df[columns[1]]):.4f}")
    print(f"  Spearman rho: {df[columns[0]].corr(df[columns[1]], method='spearman'):.4f}")

print("\nFull-dataset Table 3 values:")
print(pd.DataFrame(rows).round(4).to_string(index=False))