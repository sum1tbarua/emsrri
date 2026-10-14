# 07 subgroups drift frontier
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


# ============================================================
# EMS-RRI REVISION ANALYSIS 09
# Forward Subgroup Uncertainty + Drift Diagnostics
#
# PRIMARY MODEL:
#   Predict constructed low-reliability EMS-RRI label
#
# TEMPORAL DESIGN:
#   2024:
#       70% training
#       15% validation
#       15% internal test
#
#   2025:
#       full forward temporal validation
#
# GOALS:
#   1. Quantify subgroup performance with bootstrap 95% CIs
#   2. Quantify 2024 -> 2025 subgroup performance changes
#   3. Examine whether problematic subgroups exhibit
#      observable prevalence/covariate shifts
#
# IMPORTANT:
# Drift diagnostics are descriptive and do NOT establish
# causal explanations for subgroup performance changes.
# ============================================================


# ============================================================
# 1. Imports
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.spatial.distance import jensenshannon

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from xgboost import XGBClassifier


# ============================================================
# 2. Paths
# ============================================================

BASE = Path(
    str(ANALYSIS_ROOT)
)

DATA_2024 = (
    BASE /
    "datasets/nemsis_2024.parquet"
)

DATA_2025 = (
    BASE /
    "datasets/nemsis_2025.parquet"
)

OUT = (
    BASE /
    "revision_analysis/outputs/"
    "09_forward_subgroup_uncertainty"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

print("Output directory:")
print(OUT)


# ============================================================
# 3. Load data
# ============================================================

df24 = pd.read_parquet(
    DATA_2024
)

df25 = pd.read_parquet(
    DATA_2025
)

print("\nDataset shapes:")
print("2024:", df24.shape)
print("2025:", df25.shape)


# ============================================================
# 4. Exact final 32-predictor model
# ============================================================

numeric_features = [
    "computed_EMSChuteTimeMin",
    "computed_EMSDispatchCenterTimeSec",
    "computed_EMSSceneResponseTimeMin",
    "computed_EMSSceneTimeMin",
    "computed_EMSSceneToPatientTimeMin",
    "computed_EMSSystemResponseTimeMin",
    "computed_EMSTotalCallTimeMin",
    "computed_EMSTransportTimeMin",
    "computed_ageinyear",
]

categorical_features = [
    "computed_NasemsoRegion",
    "computed_USCensusDivision",
    "computed_USCensusRegion",
    "computed_Urbanicity",
    "eArrest_05",
    "eDisposition_16",
    "eDisposition_17",
    "eDisposition_19",
    "eDisposition_21",
    "eDisposition_22",
    "eDisposition_28",
    "eDisposition_29",
    "eDisposition_30",
    "eDisposition_32",
    "eOutcome_01",
    "eOutcome_02",
    "ePayment_01",
    "ePayment_50",
    "eSituation_02",
    "eSituation_07",
    "eSituation_08",
    "eSituation_13",
    "eSituation_20",
]

predictors = (
    numeric_features +
    categorical_features
)

assert len(predictors) == 32


# ============================================================
# 5. Recreate corrected 2024 development split
#
# Match Analysis 01 exactly: no stratification.
# ============================================================

train24, temp24 = train_test_split(
    df24,
    test_size=0.30,
    random_state=42,
    shuffle=True
)

val24, test24 = train_test_split(
    temp24,
    test_size=0.50,
    random_state=42,
    shuffle=True
)

print("\nSplit sizes:")
print("Train:", len(train24))
print("Validation:", len(val24))
print("Internal:", len(test24))
print("2025 forward:", len(df25))


# ============================================================
# 6. Derive EMS-RRI cutoff from 2024 TRAIN ONLY
# ============================================================

RRI_CUTOFF = float(
    train24[
        "ems_rri_basic"
    ].quantile(0.20)
)

print("\nEMS-RRI cutoff:")
print(RRI_CUTOFF)

assert np.isclose(
    RRI_CUTOFF,
    81.8182
)


def make_target(
    df
):

    score = pd.to_numeric(
        df[
            "ems_rri_basic"
        ],
        errors="coerce"
    )

    return (
        score <= RRI_CUTOFF
    ).astype(int)


y_train = make_target(
    train24
).to_numpy()

y_val = make_target(
    val24
).to_numpy()

y_test = make_target(
    test24
).to_numpy()

y_forward = make_target(
    df25
).to_numpy()


# ============================================================
# 7. Prepare feature matrices
# ============================================================

def prepare_X(
    df
):

    X = df[
        predictors
    ].copy()

    for col in numeric_features:

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

    return X


X_train = prepare_X(
    train24
)

X_val = prepare_X(
    val24
)

X_test = prepare_X(
    test24
)

X_forward = prepare_X(
    df25
)


# ============================================================
# 8. Exact preprocessing/model configuration
# ============================================================

num_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True
            )
        ),
        (
            "scaler",
            RobustScaler()
        )
    ]
)

cat_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            num_pipeline,
            numeric_features
        ),
        (
            "categorical",
            cat_pipeline,
            categorical_features
        )
    ]
)

classifier = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            classifier
        )
    ]
)


# ============================================================
# 9. Fit
# ============================================================

print("\nTraining primary forward model...")

model.fit(
    X_train,
    y_train
)

print("Training complete.")


# ============================================================
# 10. Validation F1 threshold
# ============================================================

val_prob = (
    model
    .predict_proba(
        X_val
    )[:, 1]
)


threshold_rows = []

for threshold in np.arange(
    0.01,
    1.00,
    0.01
):

    pred = (
        val_prob >= threshold
    ).astype(int)

    threshold_rows.append({
        "threshold":
            round(
                float(threshold),
                2
            ),

        "f1":
            f1_score(
                y_val,
                pred,
                zero_division=0
            )
    })


threshold_table = pd.DataFrame(
    threshold_rows
)

THRESHOLD = float(
    threshold_table.loc[
        threshold_table[
            "f1"
        ].idxmax(),
        "threshold"
    ]
)

print(
    "\nSelected threshold:",
    THRESHOLD
)

assert np.isclose(
    THRESHOLD,
    0.37
)


# ============================================================
# 11. Predictions
# ============================================================

test_prob = (
    model
    .predict_proba(
        X_test
    )[:, 1]
)

forward_prob = (
    model
    .predict_proba(
        X_forward
    )[:, 1]
)


# ============================================================
# 12. Metric helper
# ============================================================

def metric_values(
    y,
    prob,
    threshold=THRESHOLD
):

    y = np.asarray(y)
    prob = np.asarray(prob)

    pred = (
        prob >= threshold
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y,
            pred,
            labels=[0, 1]
        ).ravel()
    )

    result = {
        "n":
            len(y),

        "positive_n":
            int(
                np.sum(y)
            ),

        "prevalence":
            float(
                np.mean(y)
            ),

        "accuracy":
            accuracy_score(
                y,
                pred
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y,
                pred
            ),

        "precision":
            precision_score(
                y,
                pred,
                zero_division=0
            ),

        "recall":
            recall_score(
                y,
                pred,
                zero_division=0
            ),

        "f1":
            f1_score(
                y,
                pred,
                zero_division=0
            ),

        "fnr":
            (
                fn /
                (fn + tp)
            )
            if (
                fn + tp
            ) > 0
            else np.nan,

        "fpr":
            (
                fp /
                (fp + tn)
            )
            if (
                fp + tn
            ) > 0
            else np.nan
    }

    if len(
        np.unique(y)
    ) == 2:

        result[
            "roc_auc"
        ] = roc_auc_score(
            y,
            prob
        )

        result[
            "pr_auc"
        ] = average_precision_score(
            y,
            prob
        )

    else:

        result[
            "roc_auc"
        ] = np.nan

        result[
            "pr_auc"
        ] = np.nan

    return result


# ============================================================
# 13. Full model sanity check
# ============================================================

full_internal = metric_values(
    y_test,
    test_prob
)

full_forward = metric_values(
    y_forward,
    forward_prob
)

print("\n")
print("=" * 100)
print("FULL MODEL CHECK")
print("=" * 100)

print(
    pd.DataFrame([
        {
            "dataset":
                "2024 internal",

            **full_internal
        },
        {
            "dataset":
                "2025 forward",

            **full_forward
        }
    ])
    .round(6)
    .to_string(
        index=False
    )
)


# ============================================================
# 14. Subgroup definitions
#
# These directly cover the categories highlighted in the
# original subgroup analysis.
# ============================================================

SUBGROUP_COLUMNS = {
    "NASEMSO Region":
        "computed_NasemsoRegion",

    "Census Region":
        "computed_USCensusRegion",

    "Urbanicity":
        "computed_Urbanicity"
}


def clean_group_series(
    series
):

    return (
        series
        .astype("string")
        .str.strip()
    )


# ============================================================
# 15. Point-estimate subgroup table
# ============================================================

def subgroup_point_table(
    df,
    y,
    prob,
    dataset_name
):

    rows = []

    y = np.asarray(y)
    prob = np.asarray(prob)

    for dimension, column in (
        SUBGROUP_COLUMNS.items()
    ):

        groups = clean_group_series(
            df[column]
        )

        valid_values = (
            groups
            .dropna()
            .loc[
                lambda s:
                    ~s.str.lower().isin(
                        [
                            "",
                            "nan",
                            "none"
                        ]
                    )
            ]
            .unique()
        )

        for value in sorted(
            valid_values
        ):

            mask = (
                groups == value
            ).to_numpy()

            yy = y[
                mask
            ]

            pp = prob[
                mask
            ]

            if len(
                yy
            ) < 100:
                continue

            metrics = metric_values(
                yy,
                pp
            )

            rows.append({
                "dataset":
                    dataset_name,

                "dimension":
                    dimension,

                "column":
                    column,

                "group":
                    str(value),

                **metrics
            })

    return pd.DataFrame(
        rows
    )


subgroup_points = pd.concat(
    [
        subgroup_point_table(
            test24,
            y_test,
            test_prob,
            "2024 internal"
        ),

        subgroup_point_table(
            df25,
            y_forward,
            forward_prob,
            "2025 forward"
        )
    ],
    ignore_index=True
)


# ============================================================
# 16. Bootstrap CI for subgroup performance
#
# Minimum:
#   n >= 500
#   positives >= 50
#
# 300 replicates is sufficient here while keeping runtime
# manageable across all geographic subgroups.
# ============================================================

BOOTSTRAP_REPS = 300


def bootstrap_subgroup_metrics(
    y,
    prob,
    n_bootstrap=BOOTSTRAP_REPS,
    seed=42
):

    y = np.asarray(y)
    prob = np.asarray(prob)

    n = len(y)

    rng = np.random.default_rng(
        seed
    )

    rows = []

    for b in range(
        n_bootstrap
    ):

        idx = rng.integers(
            0,
            n,
            size=n
        )

        yy = y[idx]
        pp = prob[idx]

        if len(
            np.unique(yy)
        ) < 2:
            continue

        m = metric_values(
            yy,
            pp
        )

        rows.append({
            "roc_auc":
                m[
                    "roc_auc"
                ],

            "pr_auc":
                m[
                    "pr_auc"
                ],

            "f1":
                m[
                    "f1"
                ],

            "recall":
                m[
                    "recall"
                ],

            "fnr":
                m[
                    "fnr"
                ]
        })

    return pd.DataFrame(
        rows
    )


def subgroup_ci_table(
    df,
    y,
    prob,
    dataset_name,
    seed_start
):

    rows = []

    y = np.asarray(y)
    prob = np.asarray(prob)

    seed_counter = seed_start

    for dimension, column in (
        SUBGROUP_COLUMNS.items()
    ):

        groups = clean_group_series(
            df[column]
        )

        for value in sorted(
            groups.dropna().unique()
        ):

            mask = (
                groups == value
            ).to_numpy()

            yy = y[
                mask
            ]

            pp = prob[
                mask
            ]

            if len(
                yy
            ) < 500:
                continue

            if np.sum(
                yy
            ) < 50:
                continue

            point = metric_values(
                yy,
                pp
            )

            boot = bootstrap_subgroup_metrics(
                yy,
                pp,
                n_bootstrap=
                    BOOTSTRAP_REPS,
                seed=
                    seed_counter
            )

            seed_counter += 1

            row = {
                "dataset":
                    dataset_name,

                "dimension":
                    dimension,

                "group":
                    str(value),

                "n":
                    len(yy),

                "positive_n":
                    int(
                        np.sum(
                            yy
                        )
                    ),

                "prevalence":
                    point[
                        "prevalence"
                    ]
            }

            for metric in [
                "roc_auc",
                "pr_auc",
                "f1",
                "recall",
                "fnr"
            ]:

                values = (
                    boot[
                        metric
                    ]
                    .dropna()
                )

                row[
                    metric
                ] = point[
                    metric
                ]

                row[
                    f"{metric}_lower95"
                ] = values.quantile(
                    0.025
                )

                row[
                    f"{metric}_upper95"
                ] = values.quantile(
                    0.975
                )

            rows.append(
                row
            )

            print(
                dataset_name,
                "|",
                dimension,
                "|",
                value,
                "| n=",
                len(yy)
            )

    return pd.DataFrame(
        rows
    )


print("\nBootstrapping 2024 internal subgroups...")

ci_internal = subgroup_ci_table(
    test24,
    y_test,
    test_prob,
    "2024 internal",
    seed_start=100
)


print("\nBootstrapping 2025 forward subgroups...")

ci_forward = subgroup_ci_table(
    df25,
    y_forward,
    forward_prob,
    "2025 forward",
    seed_start=1000
)


subgroup_ci = pd.concat(
    [
        ci_internal,
        ci_forward
    ],
    ignore_index=True
)


# ============================================================
# 17. Independent bootstrap CIs for temporal subgroup changes
#
# Difference:
#     2025 forward - 2024 internal
#
# Metrics emphasized:
#     ROC-AUC
#     F1
#     FNR
# ============================================================

def bootstrap_temporal_difference(
    y_old,
    p_old,
    y_new,
    p_new,
    n_bootstrap=300,
    seed=42
):

    y_old = np.asarray(
        y_old
    )

    p_old = np.asarray(
        p_old
    )

    y_new = np.asarray(
        y_new
    )

    p_new = np.asarray(
        p_new
    )

    rng = np.random.default_rng(
        seed
    )

    rows = []

    for b in range(
        n_bootstrap
    ):

        idx_old = rng.integers(
            0,
            len(y_old),
            size=len(y_old)
        )

        idx_new = rng.integers(
            0,
            len(y_new),
            size=len(y_new)
        )

        yo = y_old[
            idx_old
        ]

        po = p_old[
            idx_old
        ]

        yn = y_new[
            idx_new
        ]

        pn = p_new[
            idx_new
        ]

        if (
            len(
                np.unique(
                    yo
                )
            ) < 2
            or
            len(
                np.unique(
                    yn
                )
            ) < 2
        ):
            continue

        old_m = metric_values(
            yo,
            po
        )

        new_m = metric_values(
            yn,
            pn
        )

        rows.append({
            "delta_roc_auc":
                (
                    new_m[
                        "roc_auc"
                    ]
                    -
                    old_m[
                        "roc_auc"
                    ]
                ),

            "delta_f1":
                (
                    new_m[
                        "f1"
                    ]
                    -
                    old_m[
                        "f1"
                    ]
                ),

            "delta_fnr":
                (
                    new_m[
                        "fnr"
                    ]
                    -
                    old_m[
                        "fnr"
                    ]
                )
        })

    return pd.DataFrame(
        rows
    )


change_rows = []

seed_counter = 5000


for dimension, column in (
    SUBGROUP_COLUMNS.items()
):

    old_groups = clean_group_series(
        test24[
            column
        ]
    )

    new_groups = clean_group_series(
        df25[
            column
        ]
    )

    common_groups = sorted(
        set(
            old_groups
            .dropna()
            .unique()
        )
        &
        set(
            new_groups
            .dropna()
            .unique()
        )
    )

    for value in common_groups:

        old_mask = (
            old_groups == value
        ).to_numpy()

        new_mask = (
            new_groups == value
        ).to_numpy()

        yo = y_test[
            old_mask
        ]

        po = test_prob[
            old_mask
        ]

        yn = y_forward[
            new_mask
        ]

        pn = forward_prob[
            new_mask
        ]

        if (
            len(yo) < 500
            or len(yn) < 500
            or np.sum(yo) < 50
            or np.sum(yn) < 50
        ):
            continue

        old_point = metric_values(
            yo,
            po
        )

        new_point = metric_values(
            yn,
            pn
        )

        boot = (
            bootstrap_temporal_difference(
                yo,
                po,
                yn,
                pn,
                n_bootstrap=300,
                seed=seed_counter
            )
        )

        seed_counter += 1

        row = {
            "dimension":
                dimension,

            "group":
                str(value),

            "n_2024_internal":
                len(yo),

            "n_2025_forward":
                len(yn),

            "prevalence_2024":
                old_point[
                    "prevalence"
                ],

            "prevalence_2025":
                new_point[
                    "prevalence"
                ],

            "roc_auc_2024":
                old_point[
                    "roc_auc"
                ],

            "roc_auc_2025":
                new_point[
                    "roc_auc"
                ],

            "delta_roc_auc":
                (
                    new_point[
                        "roc_auc"
                    ]
                    -
                    old_point[
                        "roc_auc"
                    ]
                ),

            "delta_roc_auc_lower95":
                boot[
                    "delta_roc_auc"
                ].quantile(
                    0.025
                ),

            "delta_roc_auc_upper95":
                boot[
                    "delta_roc_auc"
                ].quantile(
                    0.975
                ),

            "f1_2024":
                old_point[
                    "f1"
                ],

            "f1_2025":
                new_point[
                    "f1"
                ],

            "delta_f1":
                (
                    new_point[
                        "f1"
                    ]
                    -
                    old_point[
                        "f1"
                    ]
                ),

            "delta_f1_lower95":
                boot[
                    "delta_f1"
                ].quantile(
                    0.025
                ),

            "delta_f1_upper95":
                boot[
                    "delta_f1"
                ].quantile(
                    0.975
                ),

            "fnr_2024":
                old_point[
                    "fnr"
                ],

            "fnr_2025":
                new_point[
                    "fnr"
                ],

            "delta_fnr":
                (
                    new_point[
                        "fnr"
                    ]
                    -
                    old_point[
                        "fnr"
                    ]
                ),

            "delta_fnr_lower95":
                boot[
                    "delta_fnr"
                ].quantile(
                    0.025
                ),

            "delta_fnr_upper95":
                boot[
                    "delta_fnr"
                ].quantile(
                    0.975
                )
        }

        change_rows.append(
            row
        )


temporal_changes = pd.DataFrame(
    change_rows
)


# ============================================================
# 18. Priority subgroups highlighted previously
# ============================================================

PRIORITY_GROUPS = [
    (
        "NASEMSO Region",
        "Western Plains"
    ),
    (
        "NASEMSO Region",
        "Great Lakes"
    ),
    (
        "Census Region",
        "Midwest"
    ),
    (
        "Urbanicity",
        "Frontier"
    ),
    (
        "Urbanicity",
        "Rural"
    )
]


def priority_filter(
    df
):

    mask = pd.Series(
        False,
        index=df.index
    )

    for dimension, group in (
        PRIORITY_GROUPS
    ):

        mask |= (
            df[
                "dimension"
            ]
            .astype(str)
            .str.casefold()
            .eq(
                dimension.casefold()
            )
            &
            df[
                "group"
            ]
            .astype(str)
            .str.casefold()
            .eq(
                group.casefold()
            )
        )

    return df.loc[
        mask
    ].copy()


priority_ci = (
    priority_filter(
        subgroup_ci
    )
    .sort_values(
        [
            "dimension",
            "group",
            "dataset"
        ]
    )
)

priority_changes = (
    priority_filter(
        temporal_changes
    )
    .sort_values(
        [
            "dimension",
            "group"
        ]
    )
)


print("\n")
print("=" * 120)
print("PRIORITY SUBGROUP PERFORMANCE WITH 95% CIs")
print("=" * 120)

print(
    priority_ci
    .round(6)
    .to_string(
        index=False
    )
)


print("\n")
print("=" * 120)
print("PRIORITY SUBGROUP TEMPORAL CHANGES")
print("Difference = 2025 forward - 2024 internal")
print("=" * 120)

print(
    priority_changes
    .round(6)
    .to_string(
        index=False
    )
)


# ============================================================
# 19. DRIFT DIAGNOSTICS
#
# Use complete 2024 and 2025 cohorts descriptively.
#
# These analyses characterize observable distribution changes;
# they do not establish causes of performance deterioration.
# ============================================================

df24_drift = df24.copy()
df25_drift = df25.copy()

df24_drift[
    "low_reliability"
] = make_target(
    df24_drift
)

df25_drift[
    "low_reliability"
] = make_target(
    df25_drift
)


# ============================================================
# 20. Helper to find priority group values robustly
# ============================================================

COLUMN_MAP = {
    "NASEMSO Region":
        "computed_NasemsoRegion",

    "Census Region":
        "computed_USCensusRegion",

    "Urbanicity":
        "computed_Urbanicity"
}


def resolve_value(
    series,
    requested
):

    values = (
        clean_group_series(
            series
        )
        .dropna()
        .unique()
    )

    exact = [
        value
        for value in values
        if str(
            value
        ).casefold()
        ==
        requested.casefold()
    ]

    if exact:
        return exact[0]

    contains = [
        value
        for value in values
        if requested.casefold()
        in
        str(
            value
        ).casefold()
    ]

    if contains:
        return contains[0]

    return None


# ============================================================
# 21. Priority subgroup prevalence shifts
# ============================================================

prevalence_rows = []

resolved_priority_groups = []


for dimension, requested_group in (
    PRIORITY_GROUPS
):

    column = COLUMN_MAP[
        dimension
    ]

    value24 = resolve_value(
        df24_drift[
            column
        ],
        requested_group
    )

    value25 = resolve_value(
        df25_drift[
            column
        ],
        requested_group
    )

    if (
        value24 is None
        or value25 is None
    ):

        print(
            "Could not resolve:",
            dimension,
            requested_group
        )

        continue

    mask24 = (
        clean_group_series(
            df24_drift[
                column
            ]
        )
        ==
        value24
    )

    mask25 = (
        clean_group_series(
            df25_drift[
                column
            ]
        )
        ==
        value25
    )

    sub24 = df24_drift.loc[
        mask24
    ]

    sub25 = df25_drift.loc[
        mask25
    ]

    prevalence_rows.append({

        "dimension":
            dimension,

        "group":
            requested_group,

        "n_2024":
            len(
                sub24
            ),

        "n_2025":
            len(
                sub25
            ),

        "low_reliability_prevalence_2024":
            sub24[
                "low_reliability"
            ].mean(),

        "low_reliability_prevalence_2025":
            sub25[
                "low_reliability"
            ].mean(),

        "prevalence_change":
            (
                sub25[
                    "low_reliability"
                ].mean()
                -
                sub24[
                    "low_reliability"
                ].mean()
            )
    })

    resolved_priority_groups.append(
        (
            dimension,
            requested_group,
            column,
            value24,
            value25
        )
    )


prevalence_shift = pd.DataFrame(
    prevalence_rows
)

print("\n")
print("=" * 110)
print("PRIORITY SUBGROUP LOW-RELIABILITY PREVALENCE SHIFT")
print("=" * 110)

print(
    prevalence_shift
    .round(6)
    .to_string(
        index=False
    )
)


# ============================================================
# 22. Numeric PSI
# ============================================================

def numeric_psi(
    reference,
    current,
    bins=10,
    eps=1e-6
):

    ref = pd.to_numeric(
        reference,
        errors="coerce"
    )

    cur = pd.to_numeric(
        current,
        errors="coerce"
    )

    ref_nonmissing = (
        ref
        .dropna()
        .to_numpy()
    )

    cur_nonmissing = (
        cur
        .dropna()
        .to_numpy()
    )

    if (
        len(
            ref_nonmissing
        ) < 20
        or len(
            cur_nonmissing
        ) < 20
    ):
        return np.nan

    quantiles = np.linspace(
        0,
        1,
        bins + 1
    )

    edges = np.unique(
        np.quantile(
            ref_nonmissing,
            quantiles
        )
    )

    if len(
        edges
    ) < 3:
        return np.nan

    inner = edges[
        1:-1
    ]

    histogram_edges = np.concatenate(
        (
            [-np.inf],
            inner,
            [np.inf]
        )
    )

    ref_counts, _ = np.histogram(
        ref_nonmissing,
        bins=histogram_edges
    )

    cur_counts, _ = np.histogram(
        cur_nonmissing,
        bins=histogram_edges
    )

    # proportions relative to total sample so missingness
    # can be included as another bin
    ref_props = (
        ref_counts /
        len(ref)
    )

    cur_props = (
        cur_counts /
        len(cur)
    )

    ref_missing = (
        ref.isna().sum()
        /
        len(ref)
    )

    cur_missing = (
        cur.isna().sum()
        /
        len(cur)
    )

    ref_props = np.append(
        ref_props,
        ref_missing
    )

    cur_props = np.append(
        cur_props,
        cur_missing
    )

    ref_props = np.clip(
        ref_props,
        eps,
        None
    )

    cur_props = np.clip(
        cur_props,
        eps,
        None
    )

    return float(
        np.sum(
            (
                cur_props -
                ref_props
            )
            *
            np.log(
                cur_props /
                ref_props
            )
        )
    )


# ============================================================
# 23. Categorical Jensen-Shannon divergence
# ============================================================

def categorical_js(
    reference,
    current
):

    ref = (
        reference
        .astype("string")
        .fillna(
            "__MISSING__"
        )
        .str.strip()
    )

    cur = (
        current
        .astype("string")
        .fillna(
            "__MISSING__"
        )
        .str.strip()
    )

    categories = sorted(
        set(
            ref.unique()
        )
        |
        set(
            cur.unique()
        )
    )

    ref_counts = (
        ref
        .value_counts(
            normalize=True
        )
        .reindex(
            categories,
            fill_value=0
        )
        .to_numpy()
    )

    cur_counts = (
        cur
        .value_counts(
            normalize=True
        )
        .reindex(
            categories,
            fill_value=0
        )
        .to_numpy()
    )

    # squared Jensen-Shannon distance = divergence
    return float(
        jensenshannon(
            ref_counts,
            cur_counts,
            base=2
        ) ** 2
    )


# ============================================================
# 24. Predictor drift within priority subgroups
# ============================================================

drift_rows = []


for (
    dimension,
    requested_group,
    column,
    value24,
    value25
) in resolved_priority_groups:

    mask24 = (
        clean_group_series(
            df24_drift[
                column
            ]
        )
        ==
        value24
    )

    mask25 = (
        clean_group_series(
            df25_drift[
                column
            ]
        )
        ==
        value25
    )

    sub24 = df24_drift.loc[
        mask24
    ]

    sub25 = df25_drift.loc[
        mask25
    ]


    # Numeric predictors
    for feature in numeric_features:

        drift_rows.append({

            "dimension":
                dimension,

            "group":
                requested_group,

            "feature":
                feature,

            "feature_type":
                "numeric",

            "drift_metric":
                "PSI",

            "drift_value":
                numeric_psi(
                    sub24[
                        feature
                    ],
                    sub25[
                        feature
                    ]
                )
        })


    # Categorical predictors
    for feature in categorical_features:

        drift_rows.append({

            "dimension":
                dimension,

            "group":
                requested_group,

            "feature":
                feature,

            "feature_type":
                "categorical",

            "drift_metric":
                "JS divergence",

            "drift_value":
                categorical_js(
                    sub24[
                        feature
                    ],
                    sub25[
                        feature
                    ]
                )
        })


drift_df = pd.DataFrame(
    drift_rows
)


# ============================================================
# 25. Top shifts for each problematic subgroup
# ============================================================

top_shift_rows = []

for (
    dimension,
    group
) in (
    drift_df[
        [
            "dimension",
            "group"
        ]
    ]
    .drop_duplicates()
    .itertuples(
        index=False,
        name=None
    )
):

    subset = (
        drift_df[
            (
                drift_df[
                    "dimension"
                ] == dimension
            )
            &
            (
                drift_df[
                    "group"
                ] == group
            )
        ]
        .sort_values(
            "drift_value",
            ascending=False
        )
        .head(10)
        .copy()
    )

    subset[
        "rank_within_subgroup"
    ] = np.arange(
        1,
        len(
            subset
        ) + 1
    )

    top_shift_rows.append(
        subset
    )


top_shifts = pd.concat(
    top_shift_rows,
    ignore_index=True
)


print("\n")
print("=" * 120)
print("TOP OBSERVED PREDICTOR SHIFTS IN PRIORITY SUBGROUPS")
print("=" * 120)

print(
    top_shifts[
        [
            "dimension",
            "group",
            "rank_within_subgroup",
            "feature",
            "feature_type",
            "drift_metric",
            "drift_value"
        ]
    ]
    .round(6)
    .to_string(
        index=False
    )
)


# ============================================================
# 26. Save
# ============================================================

threshold_table.to_csv(
    OUT /
    "validation_threshold_search.csv",
    index=False
)

subgroup_points.to_csv(
    OUT /
    "all_subgroup_point_estimates.csv",
    index=False
)

subgroup_ci.to_csv(
    OUT /
    "all_subgroup_bootstrap_cis.csv",
    index=False
)

temporal_changes.to_csv(
    OUT /
    "subgroup_temporal_changes_with_cis.csv",
    index=False
)

priority_ci.to_csv(
    OUT /
    "priority_subgroup_bootstrap_cis.csv",
    index=False
)

priority_changes.to_csv(
    OUT /
    "priority_subgroup_temporal_changes.csv",
    index=False
)

prevalence_shift.to_csv(
    OUT /
    "priority_subgroup_prevalence_shift.csv",
    index=False
)

drift_df.to_csv(
    OUT /
    "priority_subgroup_predictor_drift.csv",
    index=False
)

top_shifts.to_csv(
    OUT /
    "priority_subgroup_top_predictor_shifts.csv",
    index=False
)

print("\nSaved outputs to:")
print(OUT)

# Next original cell

# ============================================================
# 27. Small-sample Frontier subgroup analysis
#
# Frontier was omitted from the main temporal-change analysis
# because the 2024 internal subgroup had n < 500.
#
# Since this subgroup is substantively important, estimate it
# explicitly with wider bootstrap uncertainty rather than
# silently excluding it.
# ============================================================

FRONTIER_COLUMN = "computed_Urbanicity"
FRONTIER_LABEL = "Frontier"

frontier24_series = clean_group_series(
    test24[
        FRONTIER_COLUMN
    ]
)

frontier25_series = clean_group_series(
    df25[
        FRONTIER_COLUMN
    ]
)

frontier24_mask = (
    frontier24_series
    .str.casefold()
    .eq(
        FRONTIER_LABEL.casefold()
    )
    .fillna(False)
    .to_numpy()
)

frontier25_mask = (
    frontier25_series
    .str.casefold()
    .eq(
        FRONTIER_LABEL.casefold()
    )
    .fillna(False)
    .to_numpy()
)

y_frontier24 = y_test[
    frontier24_mask
]

p_frontier24 = test_prob[
    frontier24_mask
]

y_frontier25 = y_forward[
    frontier25_mask
]

p_frontier25 = forward_prob[
    frontier25_mask
]


print("\nFrontier sample sizes:")
print(
    "2024 internal:",
    len(y_frontier24),
    "| positives:",
    int(
        y_frontier24.sum()
    )
)

print(
    "2025 forward:",
    len(y_frontier25),
    "| positives:",
    int(
        y_frontier25.sum()
    )
)


# ============================================================
# 28. Point estimates
# ============================================================

frontier24_point = metric_values(
    y_frontier24,
    p_frontier24
)

frontier25_point = metric_values(
    y_frontier25,
    p_frontier25
)


# ============================================================
# 29. Higher-replicate bootstrap for small 2024 subgroup
# ============================================================

FRONTIER_BOOTSTRAPS = 2000


frontier24_boot = bootstrap_subgroup_metrics(
    y_frontier24,
    p_frontier24,
    n_bootstrap=
        FRONTIER_BOOTSTRAPS,
    seed=9001
)

frontier25_boot = bootstrap_subgroup_metrics(
    y_frontier25,
    p_frontier25,
    n_bootstrap=
        FRONTIER_BOOTSTRAPS,
    seed=9002
)


def subgroup_boot_summary(
    dataset_name,
    point,
    boot
):

    row = {
        "dataset":
            dataset_name,

        "n":
            point[
                "n"
            ],

        "positive_n":
            point[
                "positive_n"
            ],

        "prevalence":
            point[
                "prevalence"
            ]
    }

    for metric in [
        "roc_auc",
        "pr_auc",
        "f1",
        "recall",
        "fnr"
    ]:

        values = (
            boot[
                metric
            ]
            .dropna()
        )

        row[
            metric
        ] = point[
            metric
        ]

        row[
            f"{metric}_lower95"
        ] = values.quantile(
            0.025
        )

        row[
            f"{metric}_upper95"
        ] = values.quantile(
            0.975
        )

    return row


frontier_performance = pd.DataFrame([
    subgroup_boot_summary(
        "2024 internal",
        frontier24_point,
        frontier24_boot
    ),

    subgroup_boot_summary(
        "2025 forward",
        frontier25_point,
        frontier25_boot
    )
])


print("\n")
print("=" * 115)
print("FRONTIER SMALL-SAMPLE PERFORMANCE WITH 95% CIs")
print("=" * 115)

print(
    frontier_performance
    .round(6)
    .to_string(
        index=False
    )
)


# ============================================================
# 30. Bootstrap temporal Frontier difference
#
# Independent bootstrap because cohorts are independent.
# ============================================================

frontier_change_boot = (
    bootstrap_temporal_difference(
        y_frontier24,
        p_frontier24,
        y_frontier25,
        p_frontier25,
        n_bootstrap=
            FRONTIER_BOOTSTRAPS,
        seed=9010
    )
)


frontier_change = pd.DataFrame([
    {
        "dimension":
            "Urbanicity",

        "group":
            "Frontier",

        "n_2024_internal":
            len(
                y_frontier24
            ),

        "n_2025_forward":
            len(
                y_frontier25
            ),

        "prevalence_2024":
            frontier24_point[
                "prevalence"
            ],

        "prevalence_2025":
            frontier25_point[
                "prevalence"
            ],

        "roc_auc_2024":
            frontier24_point[
                "roc_auc"
            ],

        "roc_auc_2025":
            frontier25_point[
                "roc_auc"
            ],

        "delta_roc_auc":
            (
                frontier25_point[
                    "roc_auc"
                ]
                -
                frontier24_point[
                    "roc_auc"
                ]
            ),

        "delta_roc_auc_lower95":
            frontier_change_boot[
                "delta_roc_auc"
            ].quantile(
                0.025
            ),

        "delta_roc_auc_upper95":
            frontier_change_boot[
                "delta_roc_auc"
            ].quantile(
                0.975
            ),

        "f1_2024":
            frontier24_point[
                "f1"
            ],

        "f1_2025":
            frontier25_point[
                "f1"
            ],

        "delta_f1":
            (
                frontier25_point[
                    "f1"
                ]
                -
                frontier24_point[
                    "f1"
                ]
            ),

        "delta_f1_lower95":
            frontier_change_boot[
                "delta_f1"
            ].quantile(
                0.025
            ),

        "delta_f1_upper95":
            frontier_change_boot[
                "delta_f1"
            ].quantile(
                0.975
            ),

        "fnr_2024":
            frontier24_point[
                "fnr"
            ],

        "fnr_2025":
            frontier25_point[
                "fnr"
            ],

        "delta_fnr":
            (
                frontier25_point[
                    "fnr"
                ]
                -
                frontier24_point[
                    "fnr"
                ]
            ),

        "delta_fnr_lower95":
            frontier_change_boot[
                "delta_fnr"
            ].quantile(
                0.025
            ),

        "delta_fnr_upper95":
            frontier_change_boot[
                "delta_fnr"
            ].quantile(
                0.975
            )
    }
])


print("\n")
print("=" * 115)
print("FRONTIER TEMPORAL CHANGE — 2025 FORWARD MINUS 2024 INTERNAL")
print("=" * 115)

print(
    frontier_change
    .round(6)
    .to_string(
        index=False
    )
)


# ============================================================
# 31. Save Frontier outputs
# ============================================================

frontier_performance.to_csv(
    OUT /
    "frontier_small_sample_performance_cis.csv",
    index=False
)

frontier_change.to_csv(
    OUT /
    "frontier_temporal_change_cis.csv",
    index=False
)

frontier_change_boot.to_csv(
    OUT /
    "frontier_temporal_bootstrap_replicates.csv",
    index=False
)

print(
    "\nFrontier extension saved."
)