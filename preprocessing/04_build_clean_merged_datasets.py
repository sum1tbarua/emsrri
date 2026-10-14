# Adapted paths only; source: scripts_v1/04_build_clean_merged_datasets.py
from pathlib import Path
import os
WORK_ROOT = Path(os.environ.get("NEMSIS_WORK_DIR", str(Path(__file__).resolve().parents[1] / "work"))).expanduser().resolve()
# 04_build_clean_merged_datasets.py

from pathlib import Path
import gc

import pandas as pd


# ============================================================
# 1. Paths
# ============================================================

WORK_DIR = WORK_ROOT

OUTPUT_DIR = WORK_DIR / "simple_clean_merged"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


YEAR_CONFIG = {
    2025: {
        "event_file": (
            WORK_DIR
            / "samples_2025"
            / "Pub_PCRevents_500k_random.txt"
        ),
        "filtered_dir": (
            WORK_DIR
            / "filtered_500k_2025"
        ),
        "expected_rows": 500_000,
        "output_file": (
            OUTPUT_DIR
            / "nemsis_2025_clean_merged_500k.parquet"
        ),
    },

    2024: {
        "event_file": (
            WORK_DIR
            / "samples_2024"
            / "Pub_PCRevents_2024_200k_random.txt"
        ),
        "filtered_dir": (
            WORK_DIR
            / "filtered_200k_2024"
        ),
        "expected_rows": 200_000,
        "output_file": (
            OUTPUT_DIR
            / "nemsis_2024_clean_merged_200k.parquet"
        ),
    },
}


KEY_COLUMN = "PcrKey"


# ============================================================
# 2. Special NEMSIS values observed in the audit
# ============================================================

SPECIAL_VALUE_MAP = {
    "7701001": "NOT_APPLICABLE",
    "Not Applicable": "NOT_APPLICABLE",

    "7701003": "NOT_RECORDED",
    "Not Recorded": "NOT_RECORDED",

    "8801005": "PN_EXAM_FINDING_NOT_PRESENT",
    "8801019": "PN_REFUSED",
    "8801023": "PN_UNABLE_TO_COMPLETE",
}


PERTINENT_NEGATIVE_VALUES = {
    "PN_EXAM_FINDING_NOT_PRESENT",
    "PN_REFUSED",
    "PN_UNABLE_TO_COMPLETE",
}


STATUS_SUFFIXES = [
    "_documented",
    "_not_recorded",
    "_not_applicable",
    "_blank",
    "_pertinent_negative",
]


# ============================================================
# 3. Read one NEMSIS TXT file
# ============================================================

def read_nemsis_txt(
    file_path: Path,
) -> pd.DataFrame:
    """
    Read one NEMSIS TXT file.

    All values are initially preserved as strings.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found:\n{file_path}"
        )

    print(f"\nReading:\n{file_path}")

    df = pd.read_csv(
        file_path,
        sep=r"\s*~\|~\s*",
        engine="python",
        dtype=str,
        keep_default_na=False,
        quotechar="'",
    )

    # Clean column names.
    df.columns = [
        str(column)
        .strip()
        .strip("'")
        .strip('"')
        for column in df.columns
    ]

    # Clean string values.
    for column in df.columns:
        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.strip("'")
            .str.strip('"')
        )

    if KEY_COLUMN not in df.columns:
        raise ValueError(
            f"{KEY_COLUMN} was not found in:\n{file_path}"
        )

    print(f"Loaded shape: {df.shape}")

    return df


# ============================================================
# 4. Normalize known special values
# ============================================================

def normalize_special_values(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize equivalent numeric and text special values.

    True blanks remain blank.
    """

    for column in df.columns:

        if column == KEY_COLUMN:
            continue

        df[column] = df[column].replace(
            SPECIAL_VALUE_MAP
        )

    return df


# ============================================================
# 5. Add status columns beside original values
# ============================================================

def add_status_columns(
    df: pd.DataFrame,
    columns_to_process: list[str],
) -> pd.DataFrame:
    """
    Add five numeric status columns beside each raw field.
    """

    for column in columns_to_process:

        if column not in df.columns:
            continue

        values = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        is_blank = values.eq("")

        is_not_recorded = values.eq(
            "NOT_RECORDED"
        )

        is_not_applicable = values.eq(
            "NOT_APPLICABLE"
        )

        is_pertinent_negative = values.isin(
            PERTINENT_NEGATIVE_VALUES
        )

        is_documented = ~(
            is_blank
            | is_not_recorded
            | is_not_applicable
        )

        df[f"{column}_documented"] = (
            is_documented.astype("int8")
        )

        df[f"{column}_not_recorded"] = (
            is_not_recorded.astype("int8")
        )

        df[f"{column}_not_applicable"] = (
            is_not_applicable.astype("int8")
        )

        df[f"{column}_blank"] = (
            is_blank.astype("int8")
        )

        df[f"{column}_pertinent_negative"] = (
            is_pertinent_negative.astype("int8")
        )

    return df


# ============================================================
# 6. Correct status columns after all left merges
# ============================================================

def refresh_all_status_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Recalculate every generated status column after all merges.

    This fixes events with no matching related-table record.
    For example, an event with no primary symptom row should be:

        documented = 0
        not_recorded = 0
        not_applicable = 0
        blank = 1
        pertinent_negative = 0
    """

    documented_columns = [
        column
        for column in df.columns
        if column.endswith("_documented")
    ]

    for documented_column in documented_columns:

        raw_column = documented_column.removesuffix(
            "_documented"
        )

        if raw_column not in df.columns:
            continue

        values = (
            df[raw_column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        is_blank = values.eq("")

        is_not_recorded = values.eq(
            "NOT_RECORDED"
        )

        is_not_applicable = values.eq(
            "NOT_APPLICABLE"
        )

        is_pertinent_negative = values.isin(
            PERTINENT_NEGATIVE_VALUES
        )

        is_documented = ~(
            is_blank
            | is_not_recorded
            | is_not_applicable
        )

        df[f"{raw_column}_documented"] = (
            is_documented.astype("int8")
        )

        df[f"{raw_column}_not_recorded"] = (
            is_not_recorded.astype("int8")
        )

        df[f"{raw_column}_not_applicable"] = (
            is_not_applicable.astype("int8")
        )

        df[f"{raw_column}_blank"] = (
            is_blank.astype("int8")
        )

        df[f"{raw_column}_pertinent_negative"] = (
            is_pertinent_negative.astype("int8")
        )

    return df


# ============================================================
# 7. Validate base event table
# ============================================================

def prepare_base_events(
    df: pd.DataFrame,
    expected_rows: int,
    year: int,
) -> pd.DataFrame:
    """
    Confirm exactly one row per PcrKey.
    """

    original_rows = len(df)

    df = df.drop_duplicates().copy()

    exact_duplicates_removed = (
        original_rows - len(df)
    )

    duplicate_keys = int(
        df.loc[
            df[KEY_COLUMN].duplicated(
                keep=False
            ),
            KEY_COLUMN,
        ].nunique()
    )

    print(
        f"\n{year} exact duplicate rows removed: "
        f"{exact_duplicates_removed:,}"
    )

    print(
        f"{year} conflicting duplicate PcrKeys: "
        f"{duplicate_keys:,}"
    )

    if duplicate_keys > 0:
        raise RuntimeError(
            f"{year}: the base event table contains "
            f"{duplicate_keys:,} duplicated PcrKeys "
            f"with different rows."
        )

    if len(df) != expected_rows:
        raise RuntimeError(
            f"{year}: expected {expected_rows:,} rows, "
            f"but found {len(df):,}."
        )

    return df


# ============================================================
# 8. Prepare ComputedElements or repeated small tables
# ============================================================

def prepare_related_table(
    df: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    """
    Convert a related table into one row per PcrKey.

    If a PcrKey has multiple rows, values are expanded into
    numbered columns.
    """

    df = df.drop_duplicates().copy()

    data_columns = [
        column
        for column in df.columns
        if column != KEY_COLUMN
    ]

    df["_record_number"] = (
        df.groupby(KEY_COLUMN)
        .cumcount()
        .add(1)
    )

    maximum_records = int(
        df["_record_number"].max()
    )

    print(
        f"{prefix} maximum records per PcrKey: "
        f"{maximum_records}"
    )

    # One row per PcrKey.
    if maximum_records == 1:

        output = df.drop(
            columns="_record_number"
        )

        output = output.rename(
            columns={
                column: f"{prefix}{column}"
                for column in data_columns
            }
        )

        output[f"{prefix}row_count"] = 1

        return output

    # Multiple rows per PcrKey.
    wide_parts = []

    for column in data_columns:

        wide = df.pivot(
            index=KEY_COLUMN,
            columns="_record_number",
            values=column,
        )

        wide.columns = [
            f"{prefix}{record_number}_{column}"
            for record_number in wide.columns
        ]

        wide_parts.append(wide)

    output = pd.concat(
        wide_parts,
        axis=1,
    )

    row_counts = (
        df.groupby(KEY_COLUMN)
        .size()
        .rename(f"{prefix}row_count")
    )

    output = (
        output
        .join(row_counts)
        .reset_index()
    )

    value_columns = [
        column
        for column in output.columns
        if column not in {
            KEY_COLUMN,
            f"{prefix}row_count",
        }
    ]

    output[value_columns] = (
        output[value_columns]
        .fillna("")
    )

    return output


# ============================================================
# 9. Aggregate repeated vital records
# ============================================================

def aggregate_vitals(
    vital_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize repeated vital records to one row per PcrKey.

    No clinical plausibility ranges are applied.
    """

    vital_df = vital_df.drop_duplicates().copy()

    timestamp_column = "eVitals_01"

    if timestamp_column not in vital_df.columns:
        raise ValueError(
            f"{timestamp_column} was not found "
            f"in FACTPCRVITAL."
        )

    vital_value_columns = [
        column
        for column in vital_df.columns
        if (
            column.startswith("eVitals_")
            and column != timestamp_column
        )
    ]

    if not vital_value_columns:
        raise ValueError(
            "No eVitals value columns were found."
        )

    print(
        f"Vital value columns summarized: "
        f"{len(vital_value_columns)}"
    )

    values = (
        vital_df[vital_value_columns]
        .fillna("")
        .astype(str)
    )

    blank_mask = values.eq("")

    not_recorded_mask = values.eq(
        "NOT_RECORDED"
    )

    not_applicable_mask = values.eq(
        "NOT_APPLICABLE"
    )

    pertinent_negative_mask = values.isin(
        PERTINENT_NEGATIVE_VALUES
    )

    recorded_mask = ~(
        blank_mask
        | not_recorded_mask
        | not_applicable_mask
    )

    vital_df["_recorded_count"] = (
        recorded_mask.sum(axis=1)
    )

    vital_df["_missing_count"] = (
        blank_mask.sum(axis=1)
    )

    vital_df["_not_recorded_count"] = (
        not_recorded_mask.sum(axis=1)
    )

    vital_df["_not_applicable_count"] = (
        not_applicable_mask.sum(axis=1)
    )

    vital_df["_pertinent_negative_count"] = (
        pertinent_negative_mask.sum(axis=1)
    )

    timestamp_values = (
        vital_df[timestamp_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    timestamp_has_text = ~timestamp_values.isin(
        [
            "",
            "NOT_RECORDED",
            "NOT_APPLICABLE",
        ]
    )

    parsed_timestamps = pd.to_datetime(
        timestamp_values.where(
            timestamp_has_text
        ),
        format="%d%b%Y:%H:%M:%S",
        errors="coerce",
    )

    vital_df["_timestamp_recorded"] = (
        timestamp_has_text.astype("int8")
    )

    vital_df["_timestamp_parseable"] = (
        parsed_timestamps
        .notna()
        .astype("int8")
    )

    grouped = vital_df.groupby(
        KEY_COLUMN,
        sort=False,
    )

    output = grouped.agg(
        vital_row_count=(
            KEY_COLUMN,
            "size",
        ),

        vital_fields_recorded_count=(
            "_recorded_count",
            "sum",
        ),

        vital_fields_missing_count=(
            "_missing_count",
            "sum",
        ),

        vital_not_recorded_count=(
            "_not_recorded_count",
            "sum",
        ),

        vital_not_applicable_count=(
            "_not_applicable_count",
            "sum",
        ),

        vital_pertinent_negative_count=(
            "_pertinent_negative_count",
            "sum",
        ),

        vital_timestamp_recorded_count=(
            "_timestamp_recorded",
            "sum",
        ),

        vital_timestamp_parseable_count=(
            "_timestamp_parseable",
            "sum",
        ),
    ).reset_index()

    output["has_vital_record"] = (
        output["vital_row_count"]
        .gt(0)
        .astype("int8")
    )

    output["has_valid_vital_timestamp"] = (
        output[
            "vital_timestamp_parseable_count"
        ]
        .gt(0)
        .astype("int8")
    )

    count_columns = [
        "vital_row_count",
        "vital_fields_recorded_count",
        "vital_fields_missing_count",
        "vital_not_recorded_count",
        "vital_not_applicable_count",
        "vital_pertinent_negative_count",
        "vital_timestamp_recorded_count",
        "vital_timestamp_parseable_count",
    ]

    output[count_columns] = (
        output[count_columns]
        .astype("int32")
    )

    return output


# ============================================================
# 10. Safe one-to-one merge
# ============================================================

def safe_merge(
    base_df: pd.DataFrame,
    related_df: pd.DataFrame,
    table_name: str,
) -> pd.DataFrame:
    """
    Merge without allowing row multiplication.
    """

    rows_before = len(base_df)

    merged = base_df.merge(
        related_df,
        on=KEY_COLUMN,
        how="left",
        validate="one_to_one",
    )

    if len(merged) != rows_before:
        raise RuntimeError(
            f"{table_name} merge changed row count."
        )

    print(
        f"Merged {table_name}: "
        f"{len(merged):,} rows"
    )

    return merged


# ============================================================
# 11. Prepare one related table
# ============================================================

def load_prepare_and_merge_related_table(
    base_df: pd.DataFrame,
    file_path: Path,
    sampled_keys: set,
    prefix: str,
    table_name: str,
) -> pd.DataFrame:
    """
    Load, normalize, reshape, create statuses, and merge
    one small related table.
    """

    related_df = read_nemsis_txt(
        file_path
    )

    related_df = related_df.loc[
        related_df[KEY_COLUMN].isin(
            sampled_keys
        )
    ].copy()

    related_df = normalize_special_values(
        related_df
    )

    related_ready = prepare_related_table(
        related_df,
        prefix=prefix,
    )

    value_columns = [
        column
        for column in related_ready.columns
        if (
            column != KEY_COLUMN
            and not column.endswith("row_count")
        )
    ]

    related_ready = add_status_columns(
        related_ready,
        value_columns,
    )

    base_df = safe_merge(
        base_df,
        related_ready,
        table_name,
    )

    del related_df
    del related_ready
    gc.collect()

    return base_df


# ============================================================
# 12. Process one year
# ============================================================

def process_year(
    year: int,
    config: dict,
) -> dict:
    """
    Build one clean event-level dataset.
    """

    print("\n" + "=" * 76)
    print(f"PROCESSING NEMSIS {year}")
    print("=" * 76)

    filtered_dir = config["filtered_dir"]

    computed_file = (
        filtered_dir
        / "ComputedElements.txt"
    )

    vital_file = (
        filtered_dir
        / "FACTPCRVITAL.txt"
    )

    impression_file = (
        filtered_dir
        / "FACTPCRPRIMARYIMPRESSION.txt"
    )

    symptom_file = (
        filtered_dir
        / "FACTPCRPRIMARYSYMPTOM.txt"
    )

    # --------------------------------------------------------
    # Base event table
    # --------------------------------------------------------

    base_df = read_nemsis_txt(
        config["event_file"]
    )

    base_df = normalize_special_values(
        base_df
    )

    base_df = prepare_base_events(
        df=base_df,
        expected_rows=config["expected_rows"],
        year=year,
    )

    base_raw_columns = [
        column
        for column in base_df.columns
        if column != KEY_COLUMN
    ]

    base_df = add_status_columns(
        base_df,
        base_raw_columns,
    )

    sampled_keys = set(
        base_df[KEY_COLUMN]
    )

    # --------------------------------------------------------
    # ComputedElements
    # --------------------------------------------------------

    base_df = load_prepare_and_merge_related_table(
        base_df=base_df,
        file_path=computed_file,
        sampled_keys=sampled_keys,
        prefix="computed_",
        table_name="ComputedElements",
    )

    # --------------------------------------------------------
    # Primary impression
    # --------------------------------------------------------

    base_df = load_prepare_and_merge_related_table(
        base_df=base_df,
        file_path=impression_file,
        sampled_keys=sampled_keys,
        prefix="primary_impression_",
        table_name="Primary Impression",
    )

    # --------------------------------------------------------
    # Primary symptom
    # --------------------------------------------------------

    base_df = load_prepare_and_merge_related_table(
        base_df=base_df,
        file_path=symptom_file,
        sampled_keys=sampled_keys,
        prefix="primary_symptom_",
        table_name="Primary Symptom",
    )

    # --------------------------------------------------------
    # Vitals
    # --------------------------------------------------------

    vital_df = read_nemsis_txt(
        vital_file
    )

    vital_df = vital_df.loc[
        vital_df[KEY_COLUMN].isin(
            sampled_keys
        )
    ].copy()

    vital_df = normalize_special_values(
        vital_df
    )

    vital_ready = aggregate_vitals(
        vital_df
    )

    base_df = safe_merge(
        base_df,
        vital_ready,
        "Vitals",
    )

    del vital_df
    del vital_ready
    gc.collect()

    # --------------------------------------------------------
    # Fill missing raw related-table values
    # --------------------------------------------------------

    string_columns = (
        base_df.select_dtypes(
            include="object"
        ).columns
    )

    base_df[string_columns] = (
        base_df[string_columns]
        .fillna("")
    )

    # --------------------------------------------------------
    # Correct all status indicators after left merges
    # --------------------------------------------------------

    base_df = refresh_all_status_columns(
        base_df
    )

    # --------------------------------------------------------
    # Fill vital summary columns
    # --------------------------------------------------------

    vital_count_columns = [
        "vital_row_count",
        "vital_fields_recorded_count",
        "vital_fields_missing_count",
        "vital_not_recorded_count",
        "vital_not_applicable_count",
        "vital_pertinent_negative_count",
        "vital_timestamp_recorded_count",
        "vital_timestamp_parseable_count",
    ]

    for column in vital_count_columns:

        base_df[column] = (
            pd.to_numeric(
                base_df[column],
                errors="coerce",
            )
            .fillna(0)
            .astype("int32")
        )

    vital_binary_columns = [
        "has_vital_record",
        "has_valid_vital_timestamp",
    ]

    for column in vital_binary_columns:

        base_df[column] = (
            pd.to_numeric(
                base_df[column],
                errors="coerce",
            )
            .fillna(0)
            .astype("int8")
        )

    # --------------------------------------------------------
    # Fill related-table row counts
    # --------------------------------------------------------

    row_count_columns = [
        column
        for column in base_df.columns
        if column.endswith("row_count")
    ]

    for column in row_count_columns:

        base_df[column] = (
            pd.to_numeric(
                base_df[column],
                errors="coerce",
            )
            .fillna(0)
            .astype("int32")
        )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    expected_rows = config["expected_rows"]

    if len(base_df) != expected_rows:
        raise RuntimeError(
            f"{year}: expected {expected_rows:,} rows, "
            f"but found {len(base_df):,}."
        )

    unique_keys = int(
        base_df[KEY_COLUMN].nunique()
    )

    duplicate_keys = int(
        base_df[KEY_COLUMN]
        .duplicated()
        .sum()
    )

    if unique_keys != expected_rows:
        raise RuntimeError(
            f"{year}: unique PcrKey count is incorrect."
        )

    if duplicate_keys != 0:
        raise RuntimeError(
            f"{year}: duplicate PcrKeys remain."
        )

    # --------------------------------------------------------
    # Save final dataset
    # --------------------------------------------------------

    output_file = config["output_file"]

    base_df.to_parquet(
        output_file,
        index=False,
        compression="snappy",
    )

    print("\n" + "-" * 76)
    print(f"{year} CLEAN MERGED DATASET COMPLETED")
    print("-" * 76)

    print(f"Final shape: {base_df.shape}")
    print(f"Unique PcrKeys: {unique_keys:,}")
    print(f"Duplicate PcrKeys: {duplicate_keys:,}")
    print(f"Saved to:\n{output_file}")

    audit = {
        "year": year,
        "rows": len(base_df),
        "columns": len(base_df.columns),
        "unique_pcrkeys": unique_keys,
        "duplicate_pcrkeys": duplicate_keys,

        "events_with_vital_record": int(
            base_df[
                "has_vital_record"
            ].sum()
        ),

        "events_with_valid_vital_timestamp": int(
            base_df[
                "has_valid_vital_timestamp"
            ].sum()
        ),

        "output_file": str(output_file),
    }

    del base_df
    gc.collect()

    return audit


# ============================================================
# 13. Run both years
# ============================================================

audit_records = []

for year, config in YEAR_CONFIG.items():

    audit = process_year(
        year=year,
        config=config,
    )

    audit_records.append(
        audit
    )


# ============================================================
# 14. Save audit summary
# ============================================================

audit_df = pd.DataFrame(
    audit_records
)

audit_output = (
    OUTPUT_DIR
    / "clean_merged_dataset_audit.csv"
)

audit_df.to_csv(
    audit_output,
    index=False,
)


print("\n" + "=" * 76)
print("BOTH CLEAN MERGED DATASETS COMPLETED")
print("=" * 76)

print(
    audit_df.to_string(
        index=False
    )
)

print("\nAudit saved to:")
print(audit_output)
