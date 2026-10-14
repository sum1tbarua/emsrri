# Adapted paths only; source: scripts/01_make_random_500k_event_sample.py
from pathlib import Path
import os
WORK_ROOT = Path(os.environ.get("NEMSIS_WORK_DIR", str(Path(__file__).resolve().parents[1] / "work"))).expanduser().resolve()
import subprocess
from pathlib import Path
import numpy as np
from tqdm import tqdm


# ============================================================
# 1. Paths and settings
# ============================================================

ZIP_PATH = WORK_ROOT/ "ASCII_64 2024.zip"

OUTPUT_DIR = (
    WORK_ROOT
    / "samples_2024"
)

SOURCE_FILE = "Pub_PCRevents.txt"

# For NEMSIS 2025 
# SAMPLE_SIZE = 500_000
# RANDOM_SEED = 42
# DELIMITER = "~|~"

# For NEMSIS 2025
SAMPLE_SIZE = 200_000
RANDOM_SEED = 42
DELIMITER = "~|~"

# NEMSIS 2025
# SAMPLED_EVENTS_PATH = (
#     OUTPUT_DIR / "Pub_PCRevents_500k_random.txt"
# )

# SAMPLED_KEYS_PATH = (
#     OUTPUT_DIR / "pcrkeys_500k_random.txt"
# )
# END

# NEMSIS 2024
SAMPLED_EVENTS_PATH = (
    OUTPUT_DIR / "Pub_PCRevents_2024_200k_random.txt"
)

SAMPLED_KEYS_PATH = (
    OUTPUT_DIR / "pcrkeys_2024_200k_random.txt"
)
# END

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Helper function
# ============================================================

def clean_value(value: str) -> str:
    """
    Remove surrounding spaces and quotes.
    """
    return value.strip().strip("'").strip('"')


# ============================================================
# 3. Confirm the ZIP exists
# ============================================================

if not ZIP_PATH.exists():
    raise FileNotFoundError(f"ZIP file not found:\n{ZIP_PATH}")


# ============================================================
# 4. First pass: count all event rows
# ============================================================

print("First pass: counting rows...")

count_command = [
    "unzip",
    "-p",
    str(ZIP_PATH),
    SOURCE_FILE,
]

count_process = subprocess.Popen(
    count_command,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

if count_process.stdout is None:
    raise RuntimeError("Could not open the event file stream.")

# Read and temporarily ignore the header.
header_bytes = count_process.stdout.readline()

if not header_bytes:
    error_message = count_process.stderr.read().decode(
        "utf-8",
        errors="replace",
    )
    raise RuntimeError(
        f"Could not read the header.\n{error_message}"
    )

total_rows = 0

for _ in tqdm(count_process.stdout, unit=" rows"):
    total_rows += 1

count_process.stdout.close()
return_code = count_process.wait()

if return_code != 0:
    error_message = count_process.stderr.read().decode(
        "utf-8",
        errors="replace",
    )
    raise RuntimeError(
        f"First unzip pass failed.\n{error_message}"
    )

print(f"Total event rows: {total_rows:,}")

if total_rows < SAMPLE_SIZE:
    raise ValueError(
        f"The file contains only {total_rows:,} rows, "
        f"which is fewer than the requested {SAMPLE_SIZE:,}."
    )


# ============================================================
# 5. Randomly select 500,000 row numbers
# ============================================================

rng = np.random.default_rng(RANDOM_SEED)
'''
- choose from all row positions between 0 and total_rows - 1;
- select exactly 500,000 positions;
- do not select the same row twice.
'''
selected_row_numbers = rng.choice(
    total_rows,
    size=SAMPLE_SIZE,
    replace=False,
)

# Sorting lets the second pass move forward through the file once.
selected_row_numbers.sort()

selected_row_set = set(selected_row_numbers.tolist())

print(f"Selected row positions: {len(selected_row_set):,}")
print("First 10 selected positions:")
print(selected_row_numbers[:10])


# ============================================================
# 6. Second pass: save only selected rows
# ============================================================

print("\nSecond pass: extracting selected rows...")

extract_command = [
    "unzip",
    "-p",
    str(ZIP_PATH),
    SOURCE_FILE,
]

extract_process = subprocess.Popen(
    extract_command,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

if extract_process.stdout is None:
    raise RuntimeError("Could not reopen the event file stream.")

header_bytes = extract_process.stdout.readline()

if not header_bytes:
    error_message = extract_process.stderr.read().decode(
        "utf-8",
        errors="replace",
    )
    raise RuntimeError(
        f"Could not read the header during extraction.\n"
        f"{error_message}"
    )

header = header_bytes.decode(
    "utf-8",
    errors="replace",
).rstrip("\r\n")

column_names = [
    clean_value(column)
    for column in header.split(DELIMITER)
]

if "PcrKey" not in column_names:
    raise ValueError("PcrKey was not found in the header.")

pcrkey_index = column_names.index("PcrKey")

sampled_keys = []
saved_row_count = 0

with SAMPLED_EVENTS_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as event_output:

    event_output.write(header + "\n")

    # row_index starts at 0 because selected_row_numbers also uses 0-based positions.
    for row_index, raw_line in enumerate(
        tqdm(extract_process.stdout, total=total_rows, unit=" rows")
    ):
        if row_index not in selected_row_set:
            continue

        line = raw_line.decode(
            "utf-8",
            errors="replace",
        ).rstrip("\r\n")

        values = line.split(DELIMITER)

        if len(values) <= pcrkey_index:
            raise ValueError(
                f"Selected row {row_index} does not contain PcrKey."
            )

        pcrkey = clean_value(values[pcrkey_index])

        if not pcrkey:
            raise ValueError(
                f"Selected row {row_index} has a blank PcrKey."
            )

        event_output.write(line + "\n")
        sampled_keys.append(pcrkey)
        saved_row_count += 1


extract_process.stdout.close()
return_code = extract_process.wait()

if return_code != 0:
    error_message = extract_process.stderr.read().decode(
        "utf-8",
        errors="replace",
    )
    raise RuntimeError(
        f"Second unzip pass failed.\n{error_message}"
    )


# ============================================================
# 7. Validate and save PcrKey values
# ============================================================

if saved_row_count != SAMPLE_SIZE:
    raise ValueError(
        f"Expected {SAMPLE_SIZE:,} saved rows, "
        f"but saved {saved_row_count:,}."
    )

with SAMPLED_KEYS_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as key_output:

    for pcrkey in sampled_keys:
        key_output.write(pcrkey + "\n")


# ============================================================
# 8. Final confirmation
# ============================================================

print("\nSampling complete.")
print(f"Saved event rows: {saved_row_count:,}")
print(f"Saved keys: {len(sampled_keys):,}")
print(f"Unique keys: {len(set(sampled_keys)):,}")
print(f"Event sample:\n{SAMPLED_EVENTS_PATH}")
print(f"PcrKey list:\n{SAMPLED_KEYS_PATH}")


"""
For 2025 - 
First pass: counting rows...
63635893 rows [01:32, 685117.73 rows/s]
Total event rows: 63,635,893
Selected row positions: 500,000

Sampling complete.
Saved event rows: 500,000
Saved keys: 500,000
Unique keys: 500,000
Event sample:
/Users/sumitbarua/Documents/NEMSIS_WORK/samples/Pub_PCRevents_500k_random.txt
PcrKey list:
/Users/sumitbarua/Documents/NEMSIS_WORK/samples/pcrkeys_500k_random.txt

For 2024 - 
First pass: counting rows...
60298684 rows [01:27, 689164.81 rows/s]
Total event rows: 60,298,684
Selected row positions: 200,000

Sampling complete.
Saved event rows: 200,000
Saved keys: 200,000
Unique keys: 200,000
Event sample:
/Users/sumitbarua/Documents/NEMSIS_WORK/samples_2024/Pub_PCRevents_2024_200k_random.txt
PcrKey list:
/Users/sumitbarua/Documents/NEMSIS_WORK/samples_2024/pcrkeys_2024_200k_random.txt
"""