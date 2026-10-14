# Adapted paths only; source: scripts/02_filter_related_tables_500k.py
from pathlib import Path
import os
WORK_ROOT = Path(os.environ.get("NEMSIS_WORK_DIR", str(Path(__file__).resolve().parents[1] / "work"))).expanduser().resolve()
"""
Purpose - 
Read each large NEMSIS table directly from the ZIP, keep only rows whose PcrKey belongs 
to the previously selected 500,000 events, and save the filtered tables without 
extracting the complete source files.
"""

import subprocess
from pathlib import Path
from tqdm import tqdm


# ============================================================
# 1. Paths and settings
# ============================================================

ZIP_PATH = WORK_ROOT / "ASCII 25.zip"
WORK_DIR = WORK_ROOT

KEY_PATH = (
    WORK_DIR
    / "samples_2025"
    / "pcrkeys_500k_random.txt"
)

OUT_DIR = WORK_DIR / "filtered_500k_2025"

FILES_TO_FILTER = [
    "ComputedElements.txt",
    "FACTPCRVITAL.txt",
    "FACTPCRPRIMARYIMPRESSION.txt",
    "FACTPCRPRIMARYSYMPTOM.txt",
    "PCRPATIENTRACEGROUP.txt",
]

"""
1. ComputedElements.txt
Contains one or more computed/contextual variables related to each EMS event, such as:
- age in years;
- response-time variables;
- transport-time variables;
- urbanicity;
- Census region;
- NASEMSO region.

2. FACTPCRVITAL.txt
Contains vital-sign observations.
One EMS event can have multiple vital rows because vitals may be recorded multiple times during the call.
Therefore, this filtered file can have substantially more than 500,000 rows.

3. FACTPCRPRIMARYIMPRESSION.txt
Contains primary impression information related to each event.
Some events may have multiple rows.

4. FACTPCRPRIMARYSYMPTOM.txt
Contains primary symptom information related to each event.
Some events may have multiple rows.

5. PCRPATIENTRACEGROUP.txt
Contains race-group documentation related to each event.
Some events may have multiple race-group rows.

The code correctly filters rows; it does not assume one row per event.
"""

DELIMITER = "~|~"

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Helper function
# ============================================================

def clean_token(value: str) -> str:
    """
    Remove surrounding whitespace and single or double quotes.
    """
    return value.strip().strip("'").strip('"')


# ============================================================
# 3. Check required input files
# ============================================================

if not ZIP_PATH.exists():
    raise FileNotFoundError(
        f"NEMSIS ZIP file not found:\n{ZIP_PATH}"
    )

if not KEY_PATH.exists():
    raise FileNotFoundError(
        f"Sampled PcrKey file not found:\n{KEY_PATH}"
    )


# ============================================================
# 4. Load the selected PcrKey values
# ============================================================

with KEY_PATH.open("r", encoding="utf-8") as key_file:
    pcrkeys = {
        line.strip()
        for line in key_file
        if line.strip()
    }

print(f"Loaded {len(pcrkeys):,} unique PcrKey values")

if len(pcrkeys) != 500_000:
    raise ValueError(
        f"Expected 500,000 unique PcrKey values, "
        f"but loaded {len(pcrkeys):,}."
    )


# ============================================================
# 5. Filter each related NEMSIS table
# ============================================================

for filename in FILES_TO_FILTER:

    print(f"\nFiltering {filename}...")

    output_path = OUT_DIR / filename

    command = [
        "unzip",
        "-p",
        str(ZIP_PATH),
        filename,
    ]

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1024 * 1024,
    )

    if process.stdout is None:
        raise RuntimeError(
            f"Could not open the unzip stream for {filename}."
        )

    # --------------------------------------------------------
    # Read and validate the table header
    # --------------------------------------------------------

    header_bytes = process.stdout.readline()

    if not header_bytes:
        error_message = process.stderr.read().decode(
            "utf-8",
            errors="replace",
        )

        raise RuntimeError(
            f"Could not read the header for {filename}.\n"
            f"unzip error:\n{error_message}"
        )

    header = header_bytes.decode(
        "utf-8",
        errors="replace",
    ).rstrip("\r\n")

    header_columns = [
        clean_token(column)
        for column in header.split(DELIMITER)
    ]

    if "PcrKey" not in header_columns:
        raise ValueError(
            f"PcrKey was not found in {filename}.\n"
            f"First columns: {header_columns[:10]}"
        )

    pcrkey_index = header_columns.index("PcrKey")

    # --------------------------------------------------------
    # Scan and filter data rows
    # --------------------------------------------------------

    scanned_count = 0
    kept_count = 0
    malformed_count = 0
    matched_keys = set()

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:

        output_file.write(header + "\n")

        for raw_line in tqdm(
            process.stdout,
            unit=" rows",
        ):
            scanned_count += 1

            line = raw_line.decode(
                "utf-8",
                errors="replace",
            ).rstrip("\r\n")

            fields = line.split(DELIMITER)

            if len(fields) <= pcrkey_index:
                malformed_count += 1
                continue

            pcrkey = clean_token(fields[pcrkey_index])

            if pcrkey in pcrkeys:
                output_file.write(line + "\n")
                kept_count += 1
                matched_keys.add(pcrkey)

    # --------------------------------------------------------
    # Confirm that unzip completed successfully
    # --------------------------------------------------------

    process.stdout.close()
    return_code = process.wait()

    if return_code != 0:
        error_message = process.stderr.read().decode(
            "utf-8",
            errors="replace",
        )

        raise RuntimeError(
            f"unzip failed for {filename} "
            f"with code {return_code}.\n"
            f"{error_message}"
        )

    # --------------------------------------------------------
    # Report table coverage
    # --------------------------------------------------------

    unmatched_key_count = len(pcrkeys) - len(matched_keys)

    print(f"Completed: {filename}")
    print(f"Source rows scanned: {scanned_count:,}")
    print(f"Rows retained: {kept_count:,}")
    print(f"Unique sampled events represented: {len(matched_keys):,}")
    print(f"Sampled events with no row in this table: {unmatched_key_count:,}")
    print(f"Malformed source rows skipped: {malformed_count:,}")
    print(f"Saved to:\n{output_path}")