# Adapted paths only; source: scripts/05_filter_2024.py
from pathlib import Path
import os
WORK_ROOT = Path(os.environ.get("NEMSIS_WORK_DIR", str(Path(__file__).resolve().parents[1] / "work"))).expanduser().resolve()
import subprocess
from pathlib import Path

from tqdm import tqdm


# ============================================================
# 1. Paths and settings for NEMSIS 2024
# ============================================================

ZIP_PATH = WORK_ROOT / "ASCII_64 2024.zip"

WORK_DIR = WORK_ROOT

KEY_PATH = (
    WORK_DIR
    / "samples_2024"
    / "pcrkeys_2024_200k_random.txt"
)

OUT_DIR = WORK_DIR / "filtered_200k_2024"

FILES_TO_FILTER = [
    "ComputedElements.txt",
    "FACTPCRVITAL.txt",
    "FACTPCRPRIMARYIMPRESSION.txt",
    "FACTPCRPRIMARYSYMPTOM.txt",
    "PCRPATIENTRACEGROUP.txt",
]

DELIMITER = "~|~"

EXPECTED_KEY_COUNT = 200_000

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Clean spaces and surrounding quotation marks
# ============================================================

def clean_token(value: str) -> str:
    """
    Remove surrounding spaces and single or double quotes.
    """
    return value.strip().strip("'").strip('"')


# ============================================================
# 3. Confirm required input files exist
# ============================================================

if not ZIP_PATH.exists():
    raise FileNotFoundError(
        f"NEMSIS 2024 ZIP file was not found:\n{ZIP_PATH}"
    )

if not KEY_PATH.exists():
    raise FileNotFoundError(
        f"NEMSIS 2024 PcrKey file was not found:\n{KEY_PATH}"
    )


# ============================================================
# 4. Load the 200,000 sampled PcrKey values
# ============================================================

with KEY_PATH.open("r", encoding="utf-8") as key_file:
    pcrkeys = {
        clean_token(line)
        for line in key_file
        if line.strip()
    }

print(f"Loaded {len(pcrkeys):,} unique PcrKey values")

if len(pcrkeys) != EXPECTED_KEY_COUNT:
    raise ValueError(
        f"Expected {EXPECTED_KEY_COUNT:,} unique PcrKey values, "
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
            f"First columns found: {header_columns[:10]}"
        )

    pcrkey_index = header_columns.index("PcrKey")

    # --------------------------------------------------------
    # Prepare counters
    # --------------------------------------------------------

    scanned_count = 0
    kept_count = 0
    malformed_count = 0

    # Stores only the unique sampled PcrKeys represented
    # in the filtered output table.
    matched_keys = set()

    # --------------------------------------------------------
    # Read the source table and keep matching rows
    # --------------------------------------------------------

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

            # A malformed row does not contain enough columns
            # to reach the PcrKey field.
            if len(fields) <= pcrkey_index:
                malformed_count += 1
                continue

            pcrkey = clean_token(fields[pcrkey_index])

            # Keep the row only when it belongs to one of
            # the selected 200,000 NEMSIS 2024 events.
            if pcrkey in pcrkeys:
                output_file.write(line + "\n")
                kept_count += 1
                matched_keys.add(pcrkey)

    # --------------------------------------------------------
    # Confirm unzip completed successfully
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
    # Report coverage
    # --------------------------------------------------------

    unmatched_key_count = len(pcrkeys) - len(matched_keys)

    print(f"Completed: {filename}")
    print(f"Source rows scanned: {scanned_count:,}")
    print(f"Rows retained: {kept_count:,}")
    print(
        "Unique sampled events represented: "
        f"{len(matched_keys):,}"
    )
    print(
        "Sampled events with no row in this table: "
        f"{unmatched_key_count:,}"
    )
    print(
        "Malformed source rows skipped: "
        f"{malformed_count:,}"
    )
    print(f"Saved to:\n{output_path}")