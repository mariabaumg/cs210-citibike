"""
data_prep.py - Citi Bike NYC 2022 data pipeline

Reads 12 monthly Citi Bike CSVs from data/raw/ and produces a single
aggregated Parquet file ready for ML modeling.

Usage:
    python src/data_prep.py

Input:  data/raw/2022XX-citibike-tripdata.csv  (12 files)
Output: data/processed/citibike_2022_master.parquet

The output table has one row per (station_id, month, day, hour) with these
columns.
Expected cloumns:
    station_id, lat, long,
    month, day, day_of_week, hour, is_weekend,
    in_flow, out_flow,
    lag_1h_inflow, lag_24h_inflow, lag_1h_outflow, lag_24h_outflow

Author: Maggie
"""

from pathlib import Path
import pandas as pd


# path relative to this file
REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "citibike_2022_master.parquet"

# constants
COLS_TO_LOAD = [
    "started_at", "ended_at",
    "start_station_id", "end_station_id",
    "start_lat", "start_lng",
    "end_lat", "end_lng",
]

# filters out trips shorter than 1 minute (likely false starts /
# rebalancing) or longer than 3 hours (likely forgotten / lost bikes).
MIN_DURATION_MIN = 1
MAX_DURATION_MIN = 180

YEAR = 2022


# clean one month's data
def clean_month(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(
        filepath,
        usecols=COLS_TO_LOAD,
        parse_dates=["started_at", "ended_at"],
    )

    # drop missing columns
    df = df.dropna(subset=[
        "start_station_id", "end_station_id",
        "started_at", "ended_at",
    ])

    # filter trips by duration
    duration_min = (df["ended_at"] - df["started_at"]).dt.total_seconds() / 60
    df = df[(duration_min >= MIN_DURATION_MIN) & (duration_min <= MAX_DURATION_MIN)]

    # cast station_id to string 
    df["start_station_id"] = df["start_station_id"].astype(str)
    df["end_station_id"] = df["end_station_id"].astype(str)

    return df


# aggregate trips into hourly inflow/outflow per station (station, month, day, hour)
def aggregate_month(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # time features 
    df["start_month"] = df["started_at"].dt.month
    df["start_day"] = df["started_at"].dt.day
    df["start_hour"] = df["started_at"].dt.hour
    df["end_month"] = df["ended_at"].dt.month
    df["end_day"] = df["ended_at"].dt.day
    df["end_hour"] = df["ended_at"].dt.hour

    # count trips starting at each station per hour
    outflow = (
        df.groupby(["start_station_id", "start_month", "start_day", "start_hour"])
        .size()
        .reset_index(name="out_flow")
        .rename(columns={
            "start_station_id": "station_id",
            "start_month": "month",
            "start_day": "day",
            "start_hour": "hour",
        })
    )

    # count trips ending at each station per hour
    inflow = (
        df.groupby(["end_station_id", "end_month", "end_day", "end_hour"])
        .size()
        .reset_index(name="in_flow")
        .rename(columns={
            "end_station_id": "station_id",
            "end_month": "month",
            "end_day": "day",
            "end_hour": "hour",
        })
    )

    # join station hours
    flow = outflow.merge(
        inflow,
        on=["station_id", "month", "day", "hour"],
        how="outer",
    )
    flow[["in_flow", "out_flow"]] = flow[["in_flow", "out_flow"]].fillna(0).astype(int)

    return flow


# station look up
def build_station_coords(df: pd.DataFrame) -> pd.DataFrame:
    starts = df[["start_station_id", "start_lat", "start_lng"]].rename(columns={
        "start_station_id": "station_id",
        "start_lat": "lat",
        "start_lng": "long",
    })
    ends = df[["end_station_id", "end_lat", "end_lng"]].rename(columns={
        "end_station_id": "station_id",
        "end_lat": "lat",
        "end_lng": "long",
    })
    coords = pd.concat([starts, ends], ignore_index=True)
    return coords.groupby("station_id")[["lat", "long"]].mean().reset_index()


# add time and calendar features
def add_time_and_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    # (year, month, day). 0=Monday, 6=Sunday.
    df["day_of_week"] = pd.to_datetime({
        "year": YEAR,
        "month": df["month"],
        "day": df["day"],
    }).dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # sort time
    df = df.sort_values(["station_id", "month", "day", "hour"]).reset_index(drop=True)

    grouped = df.groupby("station_id")
    df["lag_1h_inflow"] = grouped["in_flow"].shift(1)
    df["lag_24h_inflow"] = grouped["in_flow"].shift(24)
    df["lag_1h_outflow"] = grouped["out_flow"].shift(1)
    df["lag_24h_outflow"] = grouped["out_flow"].shift(24)

    # fill in missing values
    lag_cols = ["lag_1h_inflow", "lag_24h_inflow", "lag_1h_outflow", "lag_24h_outflow"]
    df[lag_cols] = df[lag_cols].fillna(0).astype(int)

    return df


# main pipeline
def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(RAW_DIR.glob("2022*-citibike-tripdata.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSVs found in {RAW_DIR}. Did you download and unzip the Kaggle data?"
        )
    if len(csv_files) != 12:
        print(f"Warning: found {len(csv_files)} CSVs, expected 12.")

    print(f"Processing {len(csv_files)} months of trip data...\n")

    monthly_aggs = []
    monthly_coords = []

    for i, filepath in enumerate(csv_files, start=1):
        print(f"[{i}/{len(csv_files)}] {filepath.name}")
        cleaned = clean_month(filepath)
        print(f"    cleaned trips:    {len(cleaned):>10,}")

        agg = aggregate_month(cleaned)
        print(f"    station-hours:    {len(agg):>10,}")
        monthly_aggs.append(agg)

        monthly_coords.append(build_station_coords(cleaned))

    # concatenate all months into one master table
    print("\nCombining all months...")
    master = pd.concat(monthly_aggs, ignore_index=True)
    print(f"    combined rows:    {len(master):>10,}")

    # build the unified station coordinates lookup
    coords = pd.concat(monthly_coords, ignore_index=True)
    coords = coords.groupby("station_id")[["lat", "long"]].mean().reset_index()
    print(f"    unique stations:  {len(coords):>10,}")

    # include coordinates to every row
    master = master.merge(coords, on="station_id", how="left")

    # add calendar and lag features
    print("\nAdding time and lag features...")
    master = add_time_and_lag_features(master)
    print(f"    final rows:       {len(master):>10,}")

    # reorder columns for readability before saving
    column_order = [
        "station_id", "lat", "long",
        "month", "day", "day_of_week", "hour", "is_weekend",
        "in_flow", "out_flow",
        "lag_1h_inflow", "lag_24h_inflow",
        "lag_1h_outflow", "lag_24h_outflow",
    ]
    master = master[column_order]

    # save
    master.to_parquet(OUTPUT_PATH, index=False)
    file_mb = OUTPUT_PATH.stat().st_size / 1024 / 1024
    print(f"\nSaved -> {OUTPUT_PATH.relative_to(REPO_ROOT)}")
    print(f"File size: {file_mb:.1f} MB")

    # quick sanity preview
    print("\nFirst row:")
    print(master.head(1).T.to_string())
    print(f"\nDtypes:\n{master.dtypes.to_string()}")


if __name__ == "__main__":
    main()