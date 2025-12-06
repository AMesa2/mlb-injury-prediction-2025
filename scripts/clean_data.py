import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/cleaned")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

def load_savant():
    path = RAW_DIR / "savant_data.csv"  # make sure this matches your filename
    df = pd.read_csv(path)
    return df

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    # Adjust these column names if yours are slightly different
    cols_needed = ["game_date", "pitcher", "release_speed", "release_spin_rate", "pitch_type"]
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in savant_data.csv: {missing}")
    
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.dropna(subset=["release_speed", "release_spin_rate"])
    df = df[df["pitch_type"].notna()]
    return df

def main():
    df = load_savant()
    df_clean = basic_clean(df)
    out_path = CLEAN_DIR / "statcast_cleaned.csv"
    df_clean.to_csv(out_path, index=False)
    print(f"Saved cleaned data to {out_path} with {len(df_clean)} rows.")

if __name__ == "__main__":
    main()
