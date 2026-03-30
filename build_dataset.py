from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


# =========================
# CONFIG
# =========================
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "E2a"
METADATA_FILE = PROJECT_DIR / "metadata.xlsx"
OUTPUT_PARQUET = PROJECT_DIR / "air_quality_complete_dataset.parquet"


def parse_filename_info(filename: str) -> tuple[str | None, str | None]:
    """
    Extract station_id and pollutant_code from filenames like:
    SPO.IT0771A_6001_BETA_2011-09-...

    Returns
    -------
    tuple
        (station_id, pollutant_code)
    """
    pattern = r"^SPO\.(?P<station_id>[A-Z0-9]+)_(?P<pollutant_code>\d+)_"
    match = re.match(pattern, filename)
    if match:
        return match.group("station_id"), match.group("pollutant_code")
    return None, None


def load_and_combine_parquets(data_dir: Path) -> pd.DataFrame:
    """
    Reads all parquet files from data_dir and concatenates them into a single DataFrame.
    Adds station_id, pollutant_code, and source_file columns.
    """
    parquet_files = sorted(data_dir.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in {data_dir}")

    dfs: list[pd.DataFrame] = []

    for file in parquet_files:
        try:
            df = pd.read_parquet(file)
        except Exception as exc:
            print(f"[WARNING] Could not read {file.name}: {exc}")
            continue

        station_id, pollutant_code = parse_filename_info(file.name)

        df["station_id"] = station_id
        df["pollutant_code"] = pollutant_code
        df["source_file"] = file.name

        dfs.append(df)

    if not dfs:
        raise ValueError("No parquet file could be successfully read.")

    combined = pd.concat(dfs, ignore_index=True)
    return combined


def clean_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans metadata and creates a station_id key compatible with parquet filenames.
    """
    meta = metadata.copy()

    if "Sampling Point Id" not in meta.columns:
        raise KeyError(
            "'Sampling Point Id' not found in metadata.xlsx. "
            f"Available columns: {list(meta.columns)}"
        )

    meta["station_id"] = (
        meta["Sampling Point Id"]
        .astype(str)
        .str.extract(r"(IT[A-Z0-9]+)", expand=False)
        .str.strip()
    )

    meta = meta.drop_duplicates(subset="station_id")

    return meta


def main() -> None:
    print("Reading parquet files...")
    df_all = load_and_combine_parquets(DATA_DIR)
    print(f"Combined parquet shape: {df_all.shape}")

    print("Reading metadata...")
    metadata = pd.read_excel(METADATA_FILE)
    print(f"Metadata shape: {metadata.shape}")

    metadata_clean = clean_metadata(metadata)

    print("\nSample station IDs from parquet files:")
    print(df_all["station_id"].dropna().unique()[:5])

    print("\nSample station IDs from metadata:")
    print(metadata_clean["station_id"].dropna().unique()[:5])

    print("\nMerging...")
    df_final = df_all.merge(
        metadata_clean,
        on="station_id",
        how="left",
        suffixes=("", "_meta"),
    )

    print(f"Final dataset shape: {df_final.shape}")

    unmatched_share = df_final["Sampling Point Id"].isna().mean()
    print(f"Share of rows without metadata match: {unmatched_share:.2%}")

    print("\nPreview:")
    print(df_final[["station_id"]].head())

    cols_to_check = [
        c for c in [
            "station_id",
            "Air Quality Station Name",
            "Countrycode",
            "Longitude",
            "Latitude",
        ]
        if c in df_final.columns
    ]
    print(df_final[cols_to_check].head())

    print("\nSaving output...")

    object_cols = df_final.select_dtypes(include=["object"]).columns
    for col in object_cols:
        df_final[col] = df_final[col].where(df_final[col].isna(), df_final[col].astype(str))

    df_final.to_parquet(OUTPUT_PARQUET, index=False)

    print("\nDone.")
    print(f"Parquet saved to: {OUTPUT_PARQUET}")

    print("\nColumns in final dataset:")
    print(df_final.columns.tolist())


if __name__ == "__main__":
    main()