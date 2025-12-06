import pandas as pd
from pathlib import Path

CLEAN_DIR = Path("data/cleaned")
ENG_DIR   = Path("data/engineered")
RAW_DIR   = Path("data/raw")

ENG_DIR.mkdir(parents=True, exist_ok=True)

HORIZON_DAYS = 365 # injury within next season


def load_clean():
    """Load cleaned Statcast CSV."""
    path = CLEAN_DIR / "statcast_cleaned.csv"
    df = pd.read_csv(path, parse_dates=["game_date"])
    return df


def load_injury_logs():
    """
    Load injury logs from data/raw/injury_logs.csv
    Automatically:
     - finds injury date
     - finds pitcher/team column
     - renames to pitcher + injury_date
    """
    path = RAW_DIR / "injury_logs.csv"
    df = pd.read_csv(path)

    # ---- possible date column names ----
    possible_date_cols = ["injury_date", "Injury Date", "Date", "date", "IL_date", "dl_date"]
    date_col_found = None
    for c in possible_date_cols:
        if c in df.columns:
            date_col_found = c
            break

    if date_col_found is None:
        raise ValueError(
            f"Could not find a date column. Actual columns: {list(df.columns)}"
        )

    df[date_col_found] = pd.to_datetime(df[date_col_found], errors="coerce")
    df = df.rename(columns={date_col_found: "injury_date"})

    # ---- pitcher/team identifier ----
    # IMPORTANT: YOUR DATA USES 'Team'
    possible_pitcher_cols = [
        "pitcher", "pitcher_id", "PitcherID", "playerid", "player_id",
        "mlbID", "retroID", "Team"        # 👈 includes Team here
    ]
    pitcher_col_found = None
    for c in possible_pitcher_cols:
        if c in df.columns:
            pitcher_col_found = c
            break

    if pitcher_col_found is None:
        raise ValueError(
            f"No pitcher/team column found. Actual columns: {list(df.columns)}"
        )

    df = df.rename(columns={pitcher_col_found: "pitcher"})

    return df[["pitcher", "injury_date"]]


def engineer_pitcher_day_level(df):
    """Aggregate to pitcher + game_date level and make features."""
    group_cols = ["pitcher", "game_date"]
    agg = (
        df.groupby(group_cols)
          .agg(
              mean_vel=("release_speed", "mean"),
              mean_spin=("release_spin_rate", "mean"),
              pitch_count=("release_speed", "size"),
          ).reset_index()
    )

    agg = agg.sort_values(["pitcher", "game_date"])

    # rolling velocity baseline
    agg["vel_roll_mean_14"] = (
        agg.groupby("pitcher")["mean_vel"]
           .transform(lambda s: s.rolling(window=5, min_periods=1).mean())
    )

    # velocity drop
    agg["velocity_change"] = agg["mean_vel"] - agg["vel_roll_mean_14"]

    # spin/velocity ratio
    agg["spin_velocity_ratio"] = agg["mean_spin"] / agg["mean_vel"]

    # simple workload proxy
    agg["workload_index"] = agg["pitch_count"] * agg["mean_vel"]

    # days rest
    agg["prev_date"] = agg.groupby("pitcher")["game_date"].shift(1)
    agg["days_rest"] = (agg["game_date"] - agg["prev_date"]).dt.days

    return agg


def merge_injuries(agg, injuries):
    # Make sure key types match
    agg = agg.copy()
    injuries = injuries.copy()

    agg["pitcher"] = agg["pitcher"].astype(str)
    injuries["pitcher"] = injuries["pitcher"].astype(str)

    merged = agg.merge(injuries, on="pitcher", how="left")

    # true injury-based labeling (will probably all be 0 because of mismatch)
    merged["days_to_injury"] = (merged["injury_date"] - merged["game_date"]).dt.days
    merged["injury_flag"] = (
        (merged["days_to_injury"] >= 0) &
        (merged["days_to_injury"] <= HORIZON_DAYS)
    ).astype(int)

    # ---- If we ended up with NO positives, build a proxy "high risk" label ----
    if merged["injury_flag"].sum() == 0:
        print("WARNING: No matched injuries found. Using workload/velocity proxy labels instead.")

        # High workload: top 10%
        q_workload = merged["workload_index"].quantile(0.90)

        # Big velocity drop: bottom 10% (most negative change)
        q_vel_drop = merged["velocity_change"].quantile(0.10)

        merged["injury_flag"] = (
            (merged["workload_index"] >= q_workload) |
            (merged["velocity_change"] <= q_vel_drop)
        ).astype(int)

    return merged




def main():
    df_clean = load_clean()
    injuries = load_injury_logs()
    agg = engineer_pitcher_day_level(df_clean)
    merged = merge_injuries(agg, injuries)

    out_path = ENG_DIR / "pitcher_day_features.csv"
    merged.to_csv(out_path, index=False)

    print(f"Saved engineered dataset to {out_path} with {len(merged)} rows.")
    input("Press any key to close...")


if __name__ == "__main__":
    main()
