#!/usr/bin/env python3
"""
ChatGPT Model 6 and Claude Sonnet 5.5 were used to assist with code development, clean-up, and documentation

Compile interval-level outputs from
biosensor_simulation_general.py.

The simulation saves two files per Slurm task:
    <profile>_sensorXX_runYYY.pkl
        Full nested simulation result, including time-series arrays.

    <profile>_sensorXX_runYYY_intervals.csv
        Compact interval-level results.

This compiler intentionally combines the interval csv files only. The full
simulation pickle files remain separate because they contain large nested
arrays and are best loaded only when a processing script actually needs them.
"""

from pathlib import Path
import importlib.util
import pandas as pd


# SETTINGS
OUTPUT_DIR = Path("simulation_results")
SIMULATION_SCRIPT = Path("biosensor_simulation_general.py")

# True = stop if even one expected Slurm output is missing.
# False = combine what exists and print a warning.
REQUIRE_ALL_FILES = True


# LOAD CURRENT SIMULATION SETTINGS
def load_simulation_config(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find simulation script: {path}\n"
            "Run this compiler from the directory containing the simulation script."
        )

    spec = importlib.util.spec_from_file_location("reset_sim_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg = load_simulation_config(SIMULATION_SCRIPT)

PROFILE_NAME = str(cfg.PROFILE_NAME)
PROFILE_STEM = PROFILE_NAME.lower()
N_SENSORS = len(cfg.SENSORS)
N_RUNS_PER_SENSOR = int(cfg.N_RUNS_PER_SENSOR)
EXPECTED_FILES = N_SENSORS * N_RUNS_PER_SENSOR

COMBINED_PKL = Path(f"combined_{PROFILE_STEM}_intervals.pkl")
COMBINED_CSV = Path(f"combined_{PROFILE_STEM}_intervals.csv")


# FIND INTERVAL OUTPUT FILES
pattern = f"{PROFILE_STEM}_sensor*_run*_intervals.csv"
files = sorted(OUTPUT_DIR.glob(pattern))

print("======================================")
print(f"Profile:              {PROFILE_NAME}")
print(f"Sensors configured:   {N_SENSORS}")
print(f"Runs per sensor:      {N_RUNS_PER_SENSOR}")
print(f"Expected files:       {EXPECTED_FILES}")
print(f"Found files:          {len(files)}")
print(f"Search pattern:       {OUTPUT_DIR / pattern}")
print("======================================")


# CHECK COMPLETENESS USING EXPECTED FILENAMES
expected_names = {
    f"{PROFILE_STEM}_sensor{sensor_idx:02d}_run{run_idx:03d}_intervals.csv"
    for sensor_idx in range(N_SENSORS)
    for run_idx in range(N_RUNS_PER_SENSOR)
}

found_names = {path.name for path in files}
missing_names = sorted(expected_names - found_names)
unexpected_names = sorted(found_names - expected_names)

if missing_names:
    print(f"\nMissing {len(missing_names)} expected files.")
    for name in missing_names[:20]:
        print(f"  MISSING: {name}")
    if len(missing_names) > 20:
        print(f"  ... and {len(missing_names) - 20} more")

if unexpected_names:
    print(f"\nFound {len(unexpected_names)} unexpected matching files.")
    for name in unexpected_names[:20]:
        print(f"  EXTRA: {name}")
    if len(unexpected_names) > 20:
        print(f"  ... and {len(unexpected_names) - 20} more")

if missing_names and REQUIRE_ALL_FILES:
    raise RuntimeError(
        "Not all expected Slurm outputs are present. "
        "Set REQUIRE_ALL_FILES=False only if you intentionally want a partial compile."
    )

if not files:
    raise RuntimeError("No interval result files were found.")


# LOAD AND COMBINE
dfs = []

for i, file in enumerate(files, start=1):
    print(f"Loading {i}/{len(files)}: {file.name}", end="\r")
    dfs.append(pd.read_csv(file))

print("\nFinished loading files.")

df_results = pd.concat(dfs, ignore_index=True)


# SORT CONSISTENTLY
sort_columns = [
    col
    for col in ["sensor_idx", "Kd_pM", "run", "step"]
    if col in df_results.columns
]

if sort_columns:
    df_results = (
        df_results
        .sort_values(sort_columns)
        .reset_index(drop=True)
    )


# BASIC INTEGRITY CHECKS
print("\nCombined DataFrame shape:")
print(df_results.shape)

print("\nColumns:")
print(df_results.columns.tolist())

if {"sensor_idx", "run"}.issubset(df_results.columns):
    print("\nIndependent runs found per sensor:")
    print(
        df_results
        .groupby(["sensor_idx", "sensor_name", "Kd_pM"], dropna=False)["run"]
        .nunique()
    )

if {"sensor_idx", "run", "step"}.issubset(df_results.columns):
    duplicate_mask = df_results.duplicated(
        subset=["sensor_idx", "run", "step"],
        keep=False,
    )
    n_duplicates = int(duplicate_mask.sum())
    if n_duplicates:
        raise RuntimeError(
            f"Found {n_duplicates} rows belonging to duplicated "
            "(sensor_idx, run, step) combinations."
        )


# SAVE COMBINED INTERVAL DATA
df_results.to_pickle(COMBINED_PKL)
df_results.to_csv(COMBINED_CSV, index=False)

print("\nSaved combined interval results as:")
print(f"  {COMBINED_PKL}")
print(f"  {COMBINED_CSV}")

print("\nFull per-run .pkl files were left untouched in:")
print(f"  {OUTPUT_DIR}/")
print("======================================")
