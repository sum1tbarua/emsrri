# Adapted paths only; source: scripts_v1/08_calculate_ems_rri.py
from pathlib import Path
import os
WORK_ROOT = Path(os.environ.get("NEMSIS_WORK_DIR", str(Path(__file__).resolve().parents[1] / "work"))).expanduser().resolve()
# 08_calculate_ems_rri.py

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. Paths
# ============================================================

WORK_DIR = WORK_ROOT

MERGED_DIR = WORK_DIR / "simple_clean_merged"
INVENTORY_DIR = WORK_DIR / "ems_rri_field_inventory"
OUTPUT_DIR = WORK_DIR / "final_ems_rri_datasets"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


PARQUET_FILES = {
    2025: (
        MERGED_DIR
        / "nemsis_2025_clean_merged_500k.parquet"
    ),
    2024: (
        MERGED_DIR
        / "nemsis_2024_clean_merged_200k.parquet"
    ),
}


OUTPUT_FILES = {
    2025: (
        OUTPUT_DIR
        / "nemsis_2025_with_ems_rri.parquet"
    ),
    2024: (
        OUTPUT_DIR
        / "nemsis_2024_with_ems_rri.parquet"
    ),
}


RULE_OUTPUT = (
    OUTPUT_DIR
    / "ems_rri_rules_used.csv"
)

AUDIT_OUTPUT = (
    OUTPUT_DIR
    / "ems_rri_calculation_audit.csv"
)


# ============================================================
# 2. Conservative EMS-RRI rules
# ============================================================

# RAW_NOT_APPLICABLE means:
#
# - actual value or pertinent negative:
#       applicable and documented
#
# - blank or NOT_RECORDED:
#       applicable but undocumented
#
# - NOT_APPLICABLE:
#       excluded from denominator
#
# TRIGGER_IN means:
#
# - field is applicable only when the trigger column contains
#   one of the listed values.


RULES = [
    # --------------------------------------------------------
    # Mandatory core documentation
    # --------------------------------------------------------

    {
        "official_element_id": "eDispatch.01",
        "raw_column": "eDispatch_01",
        "domain": "eDispatch",
        "rule_type": "RAW_NOT_APPLICABLE",
        "trigger_column": "",
        "trigger_values": "",
    },

    {
        "official_element_id": "eDisposition.27",
        "raw_column": "eDisposition_27",
        "domain": "eDisposition",
        "rule_type": "RAW_NOT_APPLICABLE",
        "trigger_column": "",
        "trigger_values": "",
    },

    {
        "official_element_id": "eResponse.05",
        "raw_column": "eResponse_05",
        "domain": "eResponse",
        "rule_type": "RAW_NOT_APPLICABLE",
        "trigger_column": "",
        "trigger_values": "",
    },

    {
        "official_element_id": "eResponse.07",
        "raw_column": "eResponse_07",
        "domain": "eResponse",
        "rule_type": "RAW_NOT_APPLICABLE",
        "trigger_column": "",
        "trigger_values": "",
    },

    {
        "official_element_id": "eResponse.23",
        "raw_column": "eResponse_23",
        "domain": "eResponse",
        "rule_type": "RAW_NOT_APPLICABLE",
        "trigger_column": "",
        "trigger_values": "",
    },

    # --------------------------------------------------------
    # Dispatch
    # --------------------------------------------------------

    {
        "official_element_id": "eDispatch.02",
        "raw_column": "eDispatch_02",
        "domain": "eDispatch",
        "rule_type": "RAW_NOT_APPLICABLE",
        "trigger_column": "",
        "trigger_values": "",
    },

    # --------------------------------------------------------
    # Cardiac-arrest status
    # --------------------------------------------------------

    {
        "official_element_id": "eArrest.01",
        "raw_column": "eArrest_01",
        "domain": "eArrest",
        "rule_type": "RAW_NOT_APPLICABLE",
        "trigger_column": "",
        "trigger_values": "",
    },

    # Cardiac arrest Yes:
    # 3001003 = Yes, prior to EMS arrival
    # 3001005 = Yes, after EMS arrival

    {
        "official_element_id": "eArrest.02",
        "raw_column": "eArrest_02",
        "domain": "eArrest",
        "rule_type": "TRIGGER_IN",
        "trigger_column": "eArrest_01",
        "trigger_values": "3001003|3001005",
    },

    {
        "official_element_id": "eArrest.07",
        "raw_column": "eArrest_07",
        "domain": "eArrest",
        "rule_type": "TRIGGER_IN",
        "trigger_column": "eArrest_01",
        "trigger_values": "3001003|3001005",
    },

    {
        "official_element_id": "eArrest.11",
        "raw_column": "eArrest_11",
        "domain": "eArrest",
        "rule_type": "TRIGGER_IN",
        "trigger_column": "eArrest_01",
        "trigger_values": "3001003|3001005",
    },

    {
        "official_element_id": "eArrest.14",
        "raw_column": "eArrest_14",
        "domain": "eArrest",
        "rule_type": "TRIGGER_IN",
        "trigger_column": "eArrest_01",
        "trigger_values": "3001003|3001005",
    },

    {
        "official_element_id": "eArrest.16",
        "raw_column": "eArrest_16",
        "domain": "eArrest",
        "rule_type": "TRIGGER_IN",
        "trigger_column": "eArrest_01",
        "trigger_values": "3001003|3001005",
    },

    {
        "official_element_id": "eArrest.18",
        "raw_column": "eArrest_18",
        "domain": "eArrest",
        "rule_type": "TRIGGER_IN",
        "trigger_column": "eArrest_01",
        "trigger_values": "3001003|3001005",
    },

    {
        "official_element_id": "eArrest.20",
        "raw_column": "eArrest_20",
        "domain": "eArrest",
        "rule_type": "TRIGGER_IN",
        "trigger_column": "eArrest_01",
        "trigger_values": "3001003|3001005",
    },

    {
        "official_element_id": "eArrest.21",
        "raw_column": "eArrest_21",
        "domain": "eArrest",
        "rule_type": "TRIGGER_IN",
        "trigger_column": "eArrest_01",
        "trigger_values": "3001003|3001005",
    },

    {
        "official_element_id": "eArrest.22",
        "raw_column": "eArrest_22",
        "domain": "eArrest",
        "rule_type": "TRIGGER_IN",
        "trigger_column": "eArrest_01",
        "trigger_values": "3001003|3001005",
    },

    # --------------------------------------------------------
    # Scene documentation
    # --------------------------------------------------------

    {
        "official_element_id": "eScene.01",
        "raw_column": "eScene_01",
        "domain": "eScene",
        "rule_type": "RAW_NOT_APPLICABLE",
        "trigger_column": "",
        "trigger_values": "",
    },

    {
        "official_element_id": "eScene.06",
        "raw_column": "eScene_06",
        "domain": "eScene",
        "rule_type": "RAW_NOT_APPLICABLE",
        "trigger_column": "",
        "trigger_values": "",
    },

    {
        "official_element_id": "eScene.07",
        "raw_column": "eScene_07",
        "domain": "eScene",
        "rule_type": "RAW_NOT_APPLICABLE",
        "trigger_column": "",
        "trigger_values": "",
    },

    # 9923003 = Mass Casualty Incident: Yes

    {
        "official_element_id": "eScene.08",
        "raw_column": "eScene_08",
        "domain": "eScene",
        "rule_type": "TRIGGER_IN",
        "trigger_column": "eScene_07",
        "trigger_values": "9923003",
    },

    {
        "official_element_id": "eScene.09",
        "raw_column": "eScene_09",
        "domain": "eScene",
        "rule_type": "RAW_NOT_APPLICABLE",
        "trigger_column": "",
        "trigger_values": "",
    },
]


# ============================================================
# 3. Helper functions
# ============================================================

def get_documented_column(
    raw_column: str,
) -> str:
    return f"{raw_column}_documented"


def get_not_applicable_column(
    raw_column: str,
) -> str:
    return f"{raw_column}_not_applicable"


def split_trigger_values(
    value: str,
) -> set[str]:
    """
    Convert:
        3001003|3001005

    into:
        {"3001003", "3001005"}
    """

    return {
        item.strip()
        for item in str(value).split("|")
        if item.strip()
    }


def determine_applicability(
    df: pd.DataFrame,
    rule: dict,
) -> pd.Series:
    """
    Determine whether one field applies to each EMS event.
    """

    raw_column = rule["raw_column"]
    rule_type = rule["rule_type"]

    if rule_type == "RAW_NOT_APPLICABLE":

        not_applicable_column = (
            get_not_applicable_column(
                raw_column
            )
        )

        return (
            pd.to_numeric(
                df[not_applicable_column],
                errors="coerce",
            )
            .fillna(0)
            .eq(0)
        )

    if rule_type == "TRIGGER_IN":

        trigger_column = rule[
            "trigger_column"
        ]

        trigger_values = split_trigger_values(
            rule["trigger_values"]
        )

        return (
            df[trigger_column]
            .fillna("")
            .astype(str)
            .str.strip()
            .isin(trigger_values)
        )

    raise ValueError(
        f"Unknown rule type: {rule_type}"
    )


# ============================================================
# 4. Validate rules against one dataset
# ============================================================

def validate_rules(
    df: pd.DataFrame,
    year: int,
) -> None:
    """
    Confirm all raw and generated status columns exist.
    """

    missing_columns = []

    for rule in RULES:

        raw_column = rule["raw_column"]

        required_columns = [
            raw_column,
            get_documented_column(
                raw_column
            ),
            get_not_applicable_column(
                raw_column
            ),
        ]

        if rule["rule_type"] == "TRIGGER_IN":
            required_columns.append(
                rule["trigger_column"]
            )

        for column in required_columns:

            if column not in df.columns:
                missing_columns.append(
                    column
                )

    missing_columns = sorted(
        set(missing_columns)
    )

    if missing_columns:
        raise RuntimeError(
            f"{year}: required columns are missing:\n"
            + "\n".join(missing_columns)
        )


# ============================================================
# 5. Calculate EMS-RRI for one year
# ============================================================

def process_year(
    year: int,
    input_file: Path,
    output_file: Path,
) -> dict:
    """
    Calculate basic and domain-weighted EMS-RRI.
    """

    print("\n" + "=" * 76)
    print(f"CALCULATING EMS-RRI FOR {year}")
    print("=" * 76)

    print(f"Loading:\n{input_file}")

    df = pd.read_parquet(
        input_file
    )

    validate_rules(
        df,
        year,
    )

    number_of_records = len(df)

    # --------------------------------------------------------
    # Domain-balanced weights
    # --------------------------------------------------------

    rules_df = pd.DataFrame(
        RULES
    )

    domain_sizes = (
        rules_df
        .groupby("domain")
        .size()
        .to_dict()
    )

    rules_df["domain_weight"] = (
        rules_df["domain"]
        .map(
            lambda domain: (
                1.0
                / domain_sizes[domain]
            )
        )
    )

    # --------------------------------------------------------
    # Initialize event-level totals
    # --------------------------------------------------------

    basic_applicable_count = np.zeros(
        number_of_records,
        dtype=np.int16,
    )

    basic_documented_count = np.zeros(
        number_of_records,
        dtype=np.int16,
    )

    weighted_applicable_sum = np.zeros(
        number_of_records,
        dtype=np.float64,
    )

    weighted_documented_sum = np.zeros(
        number_of_records,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Apply each rule
    # --------------------------------------------------------

    rule_audit_records = []

    for _, rule in rules_df.iterrows():

        rule_dict = rule.to_dict()

        raw_column = rule[
            "raw_column"
        ]

        documented_column = (
            get_documented_column(
                raw_column
            )
        )

        applicable = determine_applicability(
            df,
            rule_dict,
        ).to_numpy(
            dtype=bool
        )

        documented = (
            pd.to_numeric(
                df[documented_column],
                errors="coerce",
            )
            .fillna(0)
            .eq(1)
            .to_numpy(
                dtype=bool
            )
        )

        documented_and_applicable = (
            applicable
            & documented
        )

        basic_applicable_count += (
            applicable.astype(
                np.int16
            )
        )

        basic_documented_count += (
            documented_and_applicable.astype(
                np.int16
            )
        )

        weight = float(
            rule["domain_weight"]
        )

        weighted_applicable_sum += (
            applicable.astype(
                np.float64
            )
            * weight
        )

        weighted_documented_sum += (
            documented_and_applicable.astype(
                np.float64
            )
            * weight
        )

        rule_audit_records.append(
            {
                "year": year,
                "official_element_id":
                    rule["official_element_id"],
                "raw_column":
                    raw_column,
                "domain":
                    rule["domain"],
                "rule_type":
                    rule["rule_type"],
                "trigger_column":
                    rule["trigger_column"],
                "trigger_values":
                    rule["trigger_values"],
                "domain_weight":
                    weight,
                "applicable_count":
                    int(applicable.sum()),
                "documented_applicable_count":
                    int(
                        documented_and_applicable.sum()
                    ),
                "applicable_pct":
                    round(
                        100
                        * applicable.mean(),
                        4,
                    ),
                "documented_when_applicable_pct":
                    round(
                        (
                            100
                            * documented_and_applicable.sum()
                            / applicable.sum()
                        )
                        if applicable.sum() > 0
                        else 0,
                        4,
                    ),
            }
        )

    # --------------------------------------------------------
    # Create score columns
    # --------------------------------------------------------

    df[
        "ems_rri_applicable_item_count"
    ] = basic_applicable_count

    df[
        "ems_rri_documented_item_count"
    ] = basic_documented_count

    df[
        "ems_rri_basic"
    ] = np.where(
        basic_applicable_count > 0,
        (
            100
            * basic_documented_count
            / basic_applicable_count
        ),
        np.nan,
    ).round(4)

    df[
        "ems_rri_applicable_weight_sum"
    ] = weighted_applicable_sum.round(8)

    df[
        "ems_rri_documented_weight_sum"
    ] = weighted_documented_sum.round(8)

    df[
        "ems_rri_domain_weighted"
    ] = np.where(
        weighted_applicable_sum > 0,
        (
            100
            * weighted_documented_sum
            / weighted_applicable_sum
        ),
        np.nan,
    ).round(4)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    invalid_basic = int(
        (
            basic_documented_count
            > basic_applicable_count
        ).sum()
    )

    invalid_weighted = int(
        (
            weighted_documented_sum
            > weighted_applicable_sum
            + 1e-10
        ).sum()
    )

    if invalid_basic > 0:
        raise RuntimeError(
            f"{year}: documented count exceeds "
            f"applicable count in "
            f"{invalid_basic:,} records."
        )

    if invalid_weighted > 0:
        raise RuntimeError(
            f"{year}: weighted documented sum "
            f"exceeds weighted applicable sum."
        )

    invalid_basic_range = int(
        (
            (
                df["ems_rri_basic"] < 0
            )
            |
            (
                df["ems_rri_basic"] > 100
            )
        ).sum()
    )

    invalid_weighted_range = int(
        (
            (
                df[
                    "ems_rri_domain_weighted"
                ] < 0
            )
            |
            (
                df[
                    "ems_rri_domain_weighted"
                ] > 100
            )
        ).sum()
    )

    if invalid_basic_range > 0:
        raise RuntimeError(
            f"{year}: basic EMS-RRI outside "
            f"the 0–100 range."
        )

    if invalid_weighted_range > 0:
        raise RuntimeError(
            f"{year}: weighted EMS-RRI outside "
            f"the 0–100 range."
        )

    # --------------------------------------------------------
    # Save dataset
    # --------------------------------------------------------

    df.to_parquet(
        output_file,
        index=False,
        compression="snappy",
    )

    rule_audit = pd.DataFrame(
        rule_audit_records
    )

    rule_audit_file = (
        OUTPUT_DIR
        / f"ems_rri_rule_audit_{year}.csv"
    )

    rule_audit.to_csv(
        rule_audit_file,
        index=False,
    )

    basic_correlation = (
        df[
            [
                "ems_rri_basic",
                "ems_rri_domain_weighted",
            ]
        ]
        .corr()
        .iloc[0, 1]
    )

    audit = {
        "year": year,
        "rows": len(df),
        "rules_used": len(RULES),
        "domains_used": rules_df[
            "domain"
        ].nunique(),
        "records_without_applicable_items": int(
            (
                basic_applicable_count == 0
            ).sum()
        ),
        "basic_score_missing": int(
            df["ems_rri_basic"]
            .isna()
            .sum()
        ),
        "weighted_score_missing": int(
            df[
                "ems_rri_domain_weighted"
            ]
            .isna()
            .sum()
        ),
        "basic_mean": round(
            df["ems_rri_basic"].mean(),
            4,
        ),
        "basic_std": round(
            df["ems_rri_basic"].std(),
            4,
        ),
        "basic_median": round(
            df["ems_rri_basic"].median(),
            4,
        ),
        "weighted_mean": round(
            df[
                "ems_rri_domain_weighted"
            ].mean(),
            4,
        ),
        "weighted_std": round(
            df[
                "ems_rri_domain_weighted"
            ].std(),
            4,
        ),
        "weighted_median": round(
            df[
                "ems_rri_domain_weighted"
            ].median(),
            4,
        ),
        "basic_weighted_correlation": round(
            basic_correlation,
            6,
        ),
        "output_file": str(
            output_file
        ),
        "rule_audit_file": str(
            rule_audit_file
        ),
    }

    print(f"Rows: {len(df):,}")
    print(f"Rules used: {len(RULES):,}")
    print(
        f"Documentation domains: "
        f"{rules_df['domain'].nunique():,}"
    )

    print("\nBasic EMS-RRI:")
    print(
        df["ems_rri_basic"].describe(
            percentiles=[
                0.01,
                0.05,
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
    )

    print("\nDomain-weighted EMS-RRI:")
    print(
        df[
            "ems_rri_domain_weighted"
        ].describe(
            percentiles=[
                0.01,
                0.05,
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
    )

    print(
        "\nBasic-weighted correlation: "
        f"{basic_correlation:.6f}"
    )

    print(f"\nSaved dataset:\n{output_file}")
    print(f"\nSaved rule audit:\n{rule_audit_file}")

    return audit


# ============================================================
# 6. Validate input files
# ============================================================

for file_path in PARQUET_FILES.values():

    if not file_path.exists():
        raise FileNotFoundError(
            f"Input dataset not found:\n"
            f"{file_path}"
        )


# ============================================================
# 7. Save exact rule configuration
# ============================================================

rules_output_df = pd.DataFrame(
    RULES
)

domain_sizes = (
    rules_output_df
    .groupby("domain")
    .size()
    .to_dict()
)

rules_output_df[
    "domain_weight"
] = (
    rules_output_df["domain"]
    .map(
        lambda domain: (
            1.0
            / domain_sizes[domain]
        )
    )
)

rules_output_df[
    "include_basic"
] = "Yes"

rules_output_df[
    "include_domain_weighted"
] = "Yes"

rules_output_df[
    "methodological_note"
] = (
    "Conservative primary rule set; fields with "
    "uncertain compound applicability were excluded."
)

rules_output_df.to_csv(
    RULE_OUTPUT,
    index=False,
)


# ============================================================
# 8. Process both years
# ============================================================

audit_records = []

for year, input_file in PARQUET_FILES.items():

    audit = process_year(
        year=year,
        input_file=input_file,
        output_file=OUTPUT_FILES[year],
    )

    audit_records.append(
        audit
    )


# ============================================================
# 9. Save final audit
# ============================================================

audit_df = pd.DataFrame(
    audit_records
)

audit_df.to_csv(
    AUDIT_OUTPUT,
    index=False,
)


print("\n" + "=" * 76)
print("EMS-RRI CALCULATION COMPLETED")
print("=" * 76)

print(
    audit_df.to_string(
        index=False
    )
)

print("\nRule configuration:")
print(RULE_OUTPUT)

print("\nCalculation audit:")
print(AUDIT_OUTPUT)
