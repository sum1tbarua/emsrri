# 08 documentation state ablation
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
# EMS-RRI REVISION ANALYSIS 10
# Scoring-Element / Predictor Proxy Audit
# + Final-32 Forward Documentation-State Ablation
#
# Goals:
#   1. Recover exact 21 EMS-RRI scoring elements
#   2. Test exact overlap with final 32 predictors
#   3. Quantify documentation/missing-state associations
#   4. Re-run documentation-state ablation using:
#        2024 development -> 2025 forward validation
#   5. If exact scoring-element overlap exists:
#        retrain without overlapping predictors
#
# This analysis directly addresses possible direct/indirect
# proxying of the constructed EMS-RRI target.
# ============================================================


# ============================================================
# 1. Imports
# ============================================================

from pathlib import Path
import json
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

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
    brier_score_loss,
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

SCORING_NOTEBOOK = REPOSITORY_ROOT / "reference/scoring_rules_reference.ipynb"

OUT = (
    BASE /
    "revision_analysis/outputs/"
    "10_proxy_leakage_documentation_ablation"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

print("Output directory:")
print(OUT)


# ============================================================
# 3. Load datasets
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
# 4. Final 32 predictors
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

final_predictors = (
    numeric_features +
    categorical_features
)

assert len(final_predictors) == 32

print(
    "\nFinal predictor count:",
    len(final_predictors)
)


# ============================================================
# 5. Recover EMS-RRI scoring elements from original notebook
#
# Expected structure from manuscript:
#   eDispatch    = 2
#   eDisposition = 1
#   eResponse    = 3
#   eArrest      = 10
#   eScene       = 5
#
# Total = 21
#
# IMPORTANT:
# We do NOT silently infer the list.
# We inspect only code associated with EMS-RRI construction.
# ============================================================

assert SCORING_NOTEBOOK.exists(), (
    f"Scoring notebook not found: "
    f"{SCORING_NOTEBOOK}"
)

with open(
    SCORING_NOTEBOOK,
    "r",
    encoding="utf-8"
) as f:

    notebook = json.load(
        f
    )


construction_markers = [
    "ems_rri_basic",
    "ems_rri_domain_weighted",
    "applicable_count",
    "documented_count",
    "scoring_rules",
    "score_rules",
    "rule_definitions",
    "applicable",
    "documented"
]


# Match either:
#     eDispatch_02
# or:
#     eDispatch.02

element_pattern = re.compile(
    r"\b("
    r"eDispatch[._]\d{2}|"
    r"eDisposition[._]\d{2}|"
    r"eResponse[._]\d{2}|"
    r"eArrest[._]\d{2}|"
    r"eScene[._]\d{2}"
    r")\b"
)


construction_cells = []

candidate_counter = Counter()


for cell_number, cell in enumerate(
    notebook.get(
        "cells",
        []
    )
):

    if cell.get(
        "cell_type"
    ) != "code":
        continue

    source = "".join(
        cell.get(
            "source",
            []
        )
    )

    source_lower = (
        source.lower()
    )

    matched_markers = [
        marker
        for marker
        in construction_markers
        if marker.lower()
        in source_lower
    ]

    if not matched_markers:
        continue

    tokens = (
        element_pattern.findall(
            source
        )
    )

    tokens = [
        token.replace(
            ".",
            "_"
        )
        for token in tokens
    ]

    if tokens:

        construction_cells.append({
            "cell_number":
                cell_number,

            "markers":
                ", ".join(
                    matched_markers
                ),

            "elements":
                sorted(
                    set(tokens)
                ),

            "source":
                source
        })

        candidate_counter.update(
            tokens
        )


candidate_elements = sorted(
    candidate_counter.keys()
)


print("\n")
print("=" * 110)
print("CANDIDATE EMS-RRI ELEMENTS FOUND IN CONSTRUCTION CODE")
print("=" * 110)

for element in candidate_elements:

    print(
        f"{element:20s}",
        "occurrences =",
        candidate_counter[
            element
        ]
    )


# ============================================================
# 6. Verify domain counts
# ============================================================

expected_counts = {
    "eDispatch": 2,
    "eDisposition": 1,
    "eResponse": 3,
    "eArrest": 10,
    "eScene": 5,
}


def element_domain(
    element
):

    return element.split(
        "_"
    )[0]


observed_counts = (
    pd.Series(
        [
            element_domain(
                e
            )
            for e in candidate_elements
        ]
    )
    .value_counts()
    .to_dict()
)


print("\nExpected counts:")
print(
    expected_counts
)

print("\nObserved candidate counts:")
print(
    observed_counts
)

print(
    "\nCandidate total:",
    len(candidate_elements)
)


verified_scoring_elements = (
    len(
        candidate_elements
    ) == 21
    and
    all(
        observed_counts.get(
            domain,
            0
        )
        ==
        expected
        for domain, expected
        in expected_counts.items()
    )
)


print(
    "\nExact 21-element structure verified:",
    verified_scoring_elements
)


# ============================================================
# 7. Hard stop if extraction is ambiguous
#
# We should NEVER automatically choose among >21 candidate
# elements because conditional-applicability code may refer
# to additional NEMSIS variables.
# ============================================================

if not verified_scoring_elements:

    print("\n" + "!" * 110)

    print(
        "AUTOMATIC EXTRACTION DID NOT PRODUCE THE "
        "EXPECTED 21-ELEMENT STRUCTURE."
    )

    print(
        "Do not interpret exact-overlap results until "
        "the scoring-element list is verified."
    )

    print(
        "The construction-cell audit has been saved "
        "for manual inspection."
    )

    print("!" * 110)


# Save construction-cell audit regardless

construction_audit_df = pd.DataFrame(
    [
        {
            "cell_number":
                row[
                    "cell_number"
                ],

            "markers":
                row[
                    "markers"
                ],

            "elements":
                " | ".join(
                    row[
                        "elements"
                    ]
                ),

            "source":
                row[
                    "source"
                ]
        }
        for row in construction_cells
    ]
)

construction_audit_df.to_csv(
    OUT /
    "scoring_construction_cell_audit.csv",
    index=False
)


# ============================================================
# 8. If exact list verified, create scoring manifest
# ============================================================

if verified_scoring_elements:

    scoring_elements = (
        candidate_elements.copy()
    )

else:

    scoring_elements = []


scoring_manifest = pd.DataFrame({
    "scoring_element":
        scoring_elements
})

if len(
    scoring_manifest
) > 0:

    scoring_manifest[
        "domain"
    ] = (
        scoring_manifest[
            "scoring_element"
        ]
        .map(
            element_domain
        )
    )

scoring_manifest.to_csv(
    OUT /
    "verified_scoring_elements.csv",
    index=False
)


# ============================================================
# 9. Exact scoring-element overlap with final 32 predictors
# ============================================================

exact_overlap = sorted(
    set(
        scoring_elements
    )
    &
    set(
        final_predictors
    )
)


print("\n")
print("=" * 110)
print("EXACT SCORING-ELEMENT / FINAL-PREDICTOR OVERLAP")
print("=" * 110)

print(
    "Number of exact overlapping predictors:",
    len(exact_overlap)
)

if exact_overlap:

    for item in exact_overlap:
        print(
            "DIRECT OVERLAP:",
            item
        )

else:

    print(
        "No exact scoring-element overlap detected."
    )


# ============================================================
# 10. Predictor-level proxy audit
#
# Documentation-state values treated as potential indirect
# documentation proxies.
# ============================================================

DOCUMENTATION_STATES = {
    "NOT_RECORDED",
    "NOT_APPLICABLE",
    "NOT_REPORTING",
}


def documentation_state_mask(
    series,
    is_numeric=False
):

    if is_numeric:

        numeric = pd.to_numeric(
            series,
            errors="coerce"
        )

        return numeric.isna()

    text = (
        series
        .astype("string")
        .str.strip()
        .str.upper()
    )

    return (
        series.isna()
        |
        text.isin(
            DOCUMENTATION_STATES
        )
        |
        text.eq("")
    )


# ============================================================
# 11. Recreate primary forward split/target
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

RRI_CUTOFF = float(
    train24[
        "ems_rri_basic"
    ].quantile(
        0.20
    )
)

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
        score <=
        RRI_CUTOFF
    ).astype(int)


train_target = make_target(
    train24
)

val_target = make_target(
    val24
)

test_target = make_target(
    test24
)

forward_target = make_target(
    df25
)


# ============================================================
# 12. Documentation-state association with low-reliability
#
# This is DESCRIPTIVE.
# A large risk difference demonstrates proxy potential,
# not causal influence.
# ============================================================

proxy_rows = []


for feature in final_predictors:

    is_numeric = (
        feature in
        numeric_features
    )

    for cohort_name, dataframe, target in [
        (
            "2024 train",
            train24,
            train_target
        ),
        (
            "2025 forward",
            df25,
            forward_target
        )
    ]:

        doc_mask = (
            documentation_state_mask(
                dataframe[
                    feature
                ],
                is_numeric=
                    is_numeric
            )
            .to_numpy()
        )

        y = (
            target
            .to_numpy()
        )

        state_n = int(
            doc_mask.sum()
        )

        nonstate_n = int(
            (~doc_mask).sum()
        )

        if state_n > 0:

            risk_state = float(
                y[
                    doc_mask
                ].mean()
            )

        else:

            risk_state = np.nan

        if nonstate_n > 0:

            risk_nonstate = float(
                y[
                    ~doc_mask
                ].mean()
            )

        else:

            risk_nonstate = np.nan


        proxy_rows.append({

            "cohort":
                cohort_name,

            "predictor":
                feature,

            "predictor_type":
                (
                    "numeric"
                    if is_numeric
                    else "categorical"
                ),

            "exact_scoring_element":
                feature
                in
                exact_overlap,

            "documentation_state_n":
                state_n,

            "documentation_state_pct":
                (
                    state_n /
                    len(
                        dataframe
                    )
                ),

            "low_reliability_rate_doc_state":
                risk_state,

            "low_reliability_rate_non_doc_state":
                risk_nonstate,

            "risk_difference":
                (
                    risk_state
                    -
                    risk_nonstate
                )
                if (
                    not np.isnan(
                        risk_state
                    )
                    and
                    not np.isnan(
                        risk_nonstate
                    )
                )
                else np.nan
        })


proxy_audit = pd.DataFrame(
    proxy_rows
)


# ============================================================
# 13. Print strongest potential documentation proxies
# ============================================================

proxy_2025 = (
    proxy_audit[
        proxy_audit[
            "cohort"
        ] == "2025 forward"
    ]
    .copy()
)

proxy_2025[
    "abs_risk_difference"
] = (
    proxy_2025[
        "risk_difference"
    ].abs()
)

proxy_2025 = (
    proxy_2025
    .sort_values(
        "abs_risk_difference",
        ascending=False
    )
)


print("\n")
print("=" * 120)
print("2025 DOCUMENTATION-STATE PROXY AUDIT")
print("=" * 120)

print(
    proxy_2025[
        [
            "predictor",
            "predictor_type",
            "exact_scoring_element",
            "documentation_state_pct",
            "low_reliability_rate_doc_state",
            "low_reliability_rate_non_doc_state",
            "risk_difference"
        ]
    ]
    .head(32)
    .round(6)
    .to_string(
        index=False
    )
)


# ============================================================
# 14. Documentation-state transformation helpers
#
# ORIGINAL:
#   preserve raw documentation states
#   numeric missing indicators retained
#
# COLLAPSED:
#   NOT_RECORDED / NOT_APPLICABLE / NOT_REPORTING
#   -> one category "DOCUMENTATION_STATE"
#   numeric missing indicators retained
#
# BLIND:
#   categorical documentation states -> NaN -> modal imputation
#   numeric missing indicators removed
#
# Blind representation is intended to suppress explicit
# documentation/missingness information, while retaining
# recorded predictor values.
# ============================================================

def transform_categorical(
    dataframe,
    columns,
    mode
):

    X = dataframe[
        columns
    ].copy()

    if mode == "original":

        return X


    for col in columns:

        s = (
            X[col]
            .astype("string")
        )

        cleaned = (
            s
            .str.strip()
            .str.upper()
        )

        state_mask = (
            X[col].isna()
            |
            cleaned.isin(
                DOCUMENTATION_STATES
            )
            |
            cleaned.eq("")
        )


        if mode == "collapsed":

            X.loc[
                state_mask,
                col
            ] = (
                "DOCUMENTATION_STATE"
            )


        elif mode == "blind":

            X.loc[
                state_mask,
                col
            ] = np.nan


        else:

            raise ValueError(
                f"Unknown mode: {mode}"
            )

    return X


def prepare_representation(
    dataframe,
    predictors,
    mode
):

    numeric = [
        c
        for c in predictors
        if c in numeric_features
    ]

    categorical = [
        c
        for c in predictors
        if c in categorical_features
    ]

    X_num = dataframe[
        numeric
    ].copy()

    for col in numeric:

        X_num[col] = pd.to_numeric(
            X_num[col],
            errors="coerce"
        )

    X_cat = transform_categorical(
        dataframe,
        categorical,
        mode
    )

    X = pd.concat(
        [
            X_num,
            X_cat
        ],
        axis=1
    )

    # Preserve requested feature ordering
    X = X[
        predictors
    ]

    return (
        X,
        numeric,
        categorical
    )


# ============================================================
# 15. Model builder
# ============================================================

def build_model(
    numeric,
    categorical,
    mode
):

    numeric_add_indicator = (
        mode != "blind"
    )


    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=
                        numeric_add_indicator
                )
            ),
            (
                "scaler",
                RobustScaler()
            )
        ]
    )


    categorical_pipeline = Pipeline(
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
                numeric_pipeline,
                numeric
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical
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


    return Pipeline(
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
# 16. Threshold selection
# ============================================================

def select_threshold(
    y,
    probability
):

    rows = []

    for threshold in np.arange(
        0.01,
        1.00,
        0.01
    ):

        prediction = (
            probability >=
            threshold
        ).astype(int)

        rows.append({
            "threshold":
                round(
                    float(
                        threshold
                    ),
                    2
                ),

            "f1":
                f1_score(
                    y,
                    prediction,
                    zero_division=0
                )
        })


    table = pd.DataFrame(
        rows
    )

    best = table.loc[
        table[
            "f1"
        ].idxmax()
    ]

    return float(
        best[
            "threshold"
        ]
    )


# ============================================================
# 17. Metric helper
# ============================================================

def evaluate(
    y,
    probability,
    threshold
):

    y = np.asarray(
        y
    )

    probability = np.asarray(
        probability
    )

    prediction = (
        probability >=
        threshold
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y,
            prediction,
            labels=[0, 1]
        ).ravel()
    )

    return {
        "accuracy":
            accuracy_score(
                y,
                prediction
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y,
                prediction
            ),

        "precision":
            precision_score(
                y,
                prediction,
                zero_division=0
            ),

        "recall":
            recall_score(
                y,
                prediction,
                zero_division=0
            ),

        "f1":
            f1_score(
                y,
                prediction,
                zero_division=0
            ),

        "roc_auc":
            roc_auc_score(
                y,
                probability
            ),

        "pr_auc":
            average_precision_score(
                y,
                probability
            ),

        "brier":
            brier_score_loss(
                y,
                probability
            ),

        "fpr":
            fp /
            (
                fp + tn
            ),

        "fnr":
            fn /
            (
                fn + tp
            )
    }


# ============================================================
# 18. Define experiments
# ============================================================

experiments = {
    "Original final-32": {
        "predictors":
            final_predictors,

        "mode":
            "original"
    },

    "Collapsed documentation states": {
        "predictors":
            final_predictors,

        "mode":
            "collapsed"
    },

    "Documentation-state blind": {
        "predictors":
            final_predictors,

        "mode":
            "blind"
    }
}


# Automatically add exact-overlap exclusion if necessary.

if len(
    exact_overlap
) > 0:

    no_overlap_predictors = [
        p
        for p in final_predictors
        if p not in exact_overlap
    ]

    experiments[
        "Exclude exact scoring-element overlap"
    ] = {
        "predictors":
            no_overlap_predictors,

        "mode":
            "original"
    }

    experiments[
        "Exclude overlap + documentation blind"
    ] = {
        "predictors":
            no_overlap_predictors,

        "mode":
            "blind"
    }


print("\nExperiments:")

for name, config in (
    experiments.items()
):

    print(
        name,
        "| predictors:",
        len(
            config[
                "predictors"
            ]
        ),
        "| mode:",
        config[
            "mode"
        ]
    )


# ============================================================
# 19. Train/evaluate all variants
# ============================================================

y_train = (
    train_target
    .to_numpy()
)

y_val = (
    val_target
    .to_numpy()
)

y_test = (
    test_target
    .to_numpy()
)

y_forward = (
    forward_target
    .to_numpy()
)


results = []

prediction_store = {}


for experiment_name, config in (
    experiments.items()
):

    print("\n" + "=" * 110)
    print(
        experiment_name
    )
    print("=" * 110)

    predictors = config[
        "predictors"
    ]

    mode = config[
        "mode"
    ]


    X_train, num, cat = (
        prepare_representation(
            train24,
            predictors,
            mode
        )
    )

    X_val, _, _ = (
        prepare_representation(
            val24,
            predictors,
            mode
        )
    )

    X_test, _, _ = (
        prepare_representation(
            test24,
            predictors,
            mode
        )
    )

    X_forward, _, _ = (
        prepare_representation(
            df25,
            predictors,
            mode
        )
    )


    pipeline = build_model(
        num,
        cat,
        mode
    )


    print(
        "Training..."
    )

    pipeline.fit(
        X_train,
        y_train
    )


    validation_probability = (
        pipeline
        .predict_proba(
            X_val
        )[:, 1]
    )


    threshold = (
        select_threshold(
            y_val,
            validation_probability
        )
    )


    print(
        "Selected threshold:",
        threshold
    )


    internal_probability = (
        pipeline
        .predict_proba(
            X_test
        )[:, 1]
    )


    forward_probability = (
        pipeline
        .predict_proba(
            X_forward
        )[:, 1]
    )


    prediction_store[
        experiment_name
    ] = {
        "threshold":
            threshold,

        "internal_probability":
            internal_probability,

        "forward_probability":
            forward_probability
    }


    for dataset_name, y, prob in [
        (
            "2024 internal test",
            y_test,
            internal_probability
        ),
        (
            "2025 forward temporal",
            y_forward,
            forward_probability
        )
    ]:

        metrics = evaluate(
            y,
            prob,
            threshold
        )

        results.append({
            "experiment":
                experiment_name,

            "representation":
                mode,

            "predictor_count":
                len(
                    predictors
                ),

            "threshold":
                threshold,

            "dataset":
                dataset_name,

            **metrics
        })


results_df = pd.DataFrame(
    results
)


print("\n")
print("=" * 130)
print("FINAL-32 PROXY / DOCUMENTATION-STATE ABLATION RESULTS")
print("=" * 130)

print(
    results_df
    .round(6)
    .to_string(
        index=False
    )
)


# ============================================================
# 20. Performance losses vs original representation
#
# Positive loss = original performs better.
# ============================================================

reference = (
    results_df[
        results_df[
            "experiment"
        ]
        ==
        "Original final-32"
    ]
    .set_index(
        "dataset"
    )
)


loss_rows = []


for _, row in (
    results_df.iterrows()
):

    if (
        row[
            "experiment"
        ]
        ==
        "Original final-32"
    ):
        continue

    ref = reference.loc[
        row[
            "dataset"
        ]
    ]

    loss_rows.append({

        "experiment":
            row[
                "experiment"
            ],

        "dataset":
            row[
                "dataset"
            ],

        "predictor_count":
            row[
                "predictor_count"
            ],

        "roc_auc_loss":
            (
                ref[
                    "roc_auc"
                ]
                -
                row[
                    "roc_auc"
                ]
            ),

        "pr_auc_loss":
            (
                ref[
                    "pr_auc"
                ]
                -
                row[
                    "pr_auc"
                ]
            ),

        "f1_loss":
            (
                ref[
                    "f1"
                ]
                -
                row[
                    "f1"
                ]
            ),

        "recall_loss":
            (
                ref[
                    "recall"
                ]
                -
                row[
                    "recall"
                ]
            ),

        "brier_increase":
            (
                row[
                    "brier"
                ]
                -
                ref[
                    "brier"
                ]
            )
    })


loss_df = pd.DataFrame(
    loss_rows
)


print("\n")
print("=" * 120)
print("PERFORMANCE LOSS RELATIVE TO ORIGINAL FINAL-32")
print("Positive loss = performance deterioration")
print("=" * 120)

print(
    loss_df
    .round(6)
    .to_string(
        index=False
    )
)


# ============================================================
# 21. Paired bootstrap CIs for 2025 forward performance losses
#
# Compare every variant against original on same records.
# ============================================================

def bootstrap_variant_difference(
    y,
    original_prob,
    variant_prob,
    original_threshold,
    variant_threshold,
    n_bootstrap=500,
    seed=42
):

    y = np.asarray(
        y
    )

    original_prob = np.asarray(
        original_prob
    )

    variant_prob = np.asarray(
        variant_prob
    )

    rng = np.random.default_rng(
        seed
    )

    n = len(
        y
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

        yy = y[
            idx
        ]

        if len(
            np.unique(
                yy
            )
        ) < 2:
            continue

        original = evaluate(
            yy,
            original_prob[
                idx
            ],
            original_threshold
        )

        variant = evaluate(
            yy,
            variant_prob[
                idx
            ],
            variant_threshold
        )


        rows.append({

            # positive = variant performs worse
            "roc_auc_loss":
                (
                    original[
                        "roc_auc"
                    ]
                    -
                    variant[
                        "roc_auc"
                    ]
                ),

            "pr_auc_loss":
                (
                    original[
                        "pr_auc"
                    ]
                    -
                    variant[
                        "pr_auc"
                    ]
                ),

            "f1_loss":
                (
                    original[
                        "f1"
                    ]
                    -
                    variant[
                        "f1"
                    ]
                ),

            "recall_loss":
                (
                    original[
                        "recall"
                    ]
                    -
                    variant[
                        "recall"
                    ]
                ),

            "brier_increase":
                (
                    variant[
                        "brier"
                    ]
                    -
                    original[
                        "brier"
                    ]
                )
        })


    return pd.DataFrame(
        rows
    )


original_store = (
    prediction_store[
        "Original final-32"
    ]
)


bootstrap_rows = []


for i, experiment_name in enumerate(
    [
        name
        for name
        in experiments.keys()
        if name
        !=
        "Original final-32"
    ]
):

    print(
        "\nBootstrapping:",
        experiment_name
    )

    variant = (
        prediction_store[
            experiment_name
        ]
    )

    boot = (
        bootstrap_variant_difference(
            y_forward,
            original_store[
                "forward_probability"
            ],
            variant[
                "forward_probability"
            ],
            original_store[
                "threshold"
            ],
            variant[
                "threshold"
            ],
            n_bootstrap=500,
            seed=100 + i
        )
    )


    for metric in [
        "roc_auc_loss",
        "pr_auc_loss",
        "f1_loss",
        "recall_loss",
        "brier_increase"
    ]:

        values = (
            boot[
                metric
            ]
            .dropna()
        )

        bootstrap_rows.append({

            "experiment":
                experiment_name,

            "metric":
                metric,

            "bootstrap_mean":
                values.mean(),

            "lower_95":
                values.quantile(
                    0.025
                ),

            "upper_95":
                values.quantile(
                    0.975
                )
        })


bootstrap_summary = pd.DataFrame(
    bootstrap_rows
)


print("\n")
print("=" * 120)
print("2025 FORWARD ABLATION PERFORMANCE-LOSS 95% CIs")
print("=" * 120)

print(
    bootstrap_summary
    .round(6)
    .to_string(
        index=False
    )
)


# ============================================================
# 22. Final overlap/proxy classification table
# ============================================================

audit_rows = []


for feature in final_predictors:

    is_numeric = (
        feature in
        numeric_features
    )

    feature_proxy = (
        proxy_2025.loc[
            proxy_2025[
                "predictor"
            ]
            ==
            feature
        ]
    )


    if len(
        feature_proxy
    ) == 1:

        doc_pct = float(
            feature_proxy.iloc[
                0
            ][
                "documentation_state_pct"
            ]
        )

        risk_diff = float(
            feature_proxy.iloc[
                0
            ][
                "risk_difference"
            ]
        )

    else:

        doc_pct = np.nan
        risk_diff = np.nan


    if feature in exact_overlap:

        classification = (
            "DIRECT_SCORING_ELEMENT_OVERLAP"
        )

    elif (
        not np.isnan(
            doc_pct
        )
        and
        doc_pct > 0
    ):

        classification = (
            "POTENTIAL_DOCUMENTATION_STATE_PROXY"
        )

    else:

        classification = (
            "CONTEXTUAL_VALUE_ONLY_OR_NO_OBSERVED_STATE"
        )


    audit_rows.append({

        "predictor":
            feature,

        "predictor_type":
            (
                "numeric"
                if is_numeric
                else "categorical"
            ),

        "exact_scoring_element":
            feature
            in
            exact_overlap,

        "documentation_state_pct_2025":
            doc_pct,

        "doc_state_target_risk_difference_2025":
            risk_diff,

        "audit_classification":
            classification
    })


final_audit = pd.DataFrame(
    audit_rows
)


print("\n")
print("=" * 120)
print("FINAL PREDICTOR PROXY CLASSIFICATION")
print("=" * 120)

print(
    final_audit
    .round(6)
    .to_string(
        index=False
    )
)


# ============================================================
# 23. Save all outputs
# ============================================================

proxy_audit.to_csv(
    OUT /
    "predictor_documentation_state_proxy_audit.csv",
    index=False
)

final_audit.to_csv(
    OUT /
    "final_predictor_proxy_classification.csv",
    index=False
)

results_df.to_csv(
    OUT /
    "documentation_state_ablation_results.csv",
    index=False
)

loss_df.to_csv(
    OUT /
    "documentation_state_ablation_performance_losses.csv",
    index=False
)

bootstrap_summary.to_csv(
    OUT /
    "documentation_state_ablation_bootstrap_cis.csv",
    index=False
)

pd.DataFrame({
    "exact_overlap":
        exact_overlap
}).to_csv(
    OUT /
    "exact_scoring_predictor_overlap.csv",
    index=False
)


# ============================================================
# 24. Compact final audit summary
# ============================================================

print("\n")
print("=" * 120)
print("ANALYSIS 10 FINAL AUDIT SUMMARY")
print("=" * 120)

print(
    "Verified 21 scoring elements:",
    verified_scoring_elements
)

print(
    "Exact scoring-element overlaps:",
    len(
        exact_overlap
    )
)

print(
    "Overlapping elements:",
    exact_overlap
)

print(
    "Final-32 predictors:",
    len(
        final_predictors
    )
)

print(
    "Potential documentation-state proxies:",
    int(
        (
            final_audit[
                "audit_classification"
            ]
            ==
            "POTENTIAL_DOCUMENTATION_STATE_PROXY"
        ).sum()
    )
)

print(
    "\nOutputs saved to:"
)

print(
    OUT
)