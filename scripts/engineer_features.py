import pandas as pd
from pathlib import Path

CLEAN_DIR = Path("data/cleaned")
ENG_DIR = Path("data/engineered")
RAW_DIR = Path("data/raw")
ENG_DIR.mkdir(parents=True, exist_ok=True)

HORIZON_DAYS = 30  # injury within next 30 days

def load_clean():
    path = CLEAN_DIR / "statcast_cleaned.csv"
    df = pd.read_csv(path, parse_dates=["game_date"])
    return df

def load_injury_logs():
    """
    Expect: data/raw/injury_logs.csv with columns at least:
      pitcher_id, injury_date
    You may need to rename columns in your actual file to match this.
    """
    path = RAW_DIR / "injury_logs.csv"
    df = pd.read_csv(path, parse_dates=["injury_date"])
    return df

def engineer_pitcher_day_level(df):
    # aggregate to pitcher + game_date level
    group_cols = ["pitcher", "game_date"]
    agg = df.groupby(group_cols).agg(
        mean_vel=("release_speed", "mean"),
        mean_spin=("release_spin_rate", "mean"),
        pitch_count=("release_speed", "size")
    ).reset_index()

    agg = agg.sort_values(["pitcher", "game_date"])

    # rolling velocity baseline over recent appearances (window=5 games)
    agg["vel_roll_mean_14"] = agg.groupby("pitcher")["mean_vel"].transform(
        lambda s: s.rolling(window=5, min_periods=1).mean()
    )
    # velocity drop compared to recent baseline
    agg["velocity_change"] = agg["mean_vel"] - agg["vel_roll_mean_14"]

    # spin / velocity ratio (strain proxy)
    agg["spin_velocity_ratio"] = agg["mean_spin"] / agg["mean_vel"]

    # simple workload proxy
    agg["workload_index"] = agg["pitch_count"] * agg["mean_vel"]

    # days of rest between appearances
    agg["prev_date"] = agg.groupby("pitcher")["game_date"].shift(1)
    agg["days_rest"] = (agg["game_date"] - agg["prev_date"]).dt.days

    return agg

def merge_injuries(agg, injuries):
    injuries = injuries.rename(columns={"pitcher_id": "pitcher"})
    injuries = injuries.sort_values(["pitcher", "injury_date"])

    merged = agg.merge(injuries, on="pitcher", how="left")
    merged["days_to_injury"] = (merged["injury_date"] - merged["game_date"]).dt.days

    merged["injury_flag"] = (
        (merged["days_to_injury"] >= 0) &
        (merged["days_to_injury"] <= HORIZON_DAYS)
    ).astype(int)

    return merged

def main():
    df = load_clean()
    injuries = load_injury_logs()
    agg = engineer_pitcher_day_level(df)
    merged = merge_injuries(agg, injuries)
    out_path = ENG_DIR / "pitcher_day_features.csv"
    merged.to_csv(out_path, index=False)
    print(f"Saved engineered dataset to {out_path} with {len(merged)} rows.")

if __name__ == "__main__":
    main()
