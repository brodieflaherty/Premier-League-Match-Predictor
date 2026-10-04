"""Download, combine and clean Premier League match data from football-data.co.uk.

Usage (from the repo root):
    python src/data_loader.py            # build data/processed/matches.csv
    python src/data_loader.py --download # also (re)download any missing raw seasons
"""

import argparse
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_PATH = ROOT / "data" / "processed" / "matches.csv"

FIRST_SEASON = 2014  # 2014/15
LAST_SEASON = 2025   # 2025/26

URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"

# Columns kept in the clean table. Everything here is either the result,
# in-match stats (only usable as *past* form, never for the match itself),
# or pre-match bookmaker odds.
COLUMNS = {
    "Date": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
    "HTHG": "home_goals_ht",
    "HTAG": "away_goals_ht",
    "Referee": "referee",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HC": "home_corners",
    "AC": "away_corners",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HY": "home_yellows",
    "AY": "away_yellows",
    "HR": "home_reds",
    "AR": "away_reds",
    # Bet365 opening odds: complete for every season.
    "B365H": "b365_home",
    "B365D": "b365_draw",
    "B365A": "b365_away",
    # Pinnacle opening and closing odds: the sharpest market, but some gaps in 2025/26.
    "PSH": "pinnacle_home",
    "PSD": "pinnacle_draw",
    "PSA": "pinnacle_away",
    "PSCH": "pinnacle_close_home",
    "PSCD": "pinnacle_close_draw",
    "PSCA": "pinnacle_close_away",
    # Market average odds.
    "AvgH": "avg_home",
    "AvgD": "avg_draw",
    "AvgA": "avg_away",
}

# Seasons up to 2018/19 use the older "BbAv" names for market-average odds.
LEGACY_NAMES = {"BbAvH": "AvgH", "BbAvD": "AvgD", "BbAvA": "AvgA"}


def season_label(start_year: int) -> str:
    """2014 -> '2014/15'"""
    return f"{start_year}/{str(start_year + 1)[-2:]}"


def raw_path(start_year: int) -> Path:
    return RAW_DIR / f"premier_league_{start_year}_{start_year + 1}.csv"


def download_seasons(overwrite: bool = False) -> None:
    """Download any missing season CSVs into data/raw/."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for year in range(FIRST_SEASON, LAST_SEASON + 1):
        path = raw_path(year)
        if path.exists() and not overwrite:
            continue
        code = f"{str(year)[-2:]}{str(year + 1)[-2:]}"  # 2014 -> "1415"
        print(f"Downloading {season_label(year)} ...")
        urlretrieve(URL.format(code=code), path)


def load_season(start_year: int) -> pd.DataFrame:
    """Load one raw season and return it with standardised columns."""
    df = pd.read_csv(raw_path(start_year), encoding="latin-1")
    df = df.dropna(how="all")  # some files have trailing blank rows
    df = df.rename(columns=LEGACY_NAMES)

    df = df[[c for c in COLUMNS if c in df.columns]].rename(columns=COLUMNS)
    df["season"] = season_label(start_year)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates, sort, and sanity-check the combined table."""
    # Older files use dd/mm/yy, newer ones dd/mm/yyyy.
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="mixed")

    for col in ["home_team", "away_team", "referee"]:
        df[col] = df[col].str.strip()

    df = df.sort_values(["date", "home_team"], kind="stable").reset_index(drop=True)
    df.insert(0, "match_id", range(len(df)))

    # Reorder so season sits next to date.
    cols = ["match_id", "season"] + [c for c in df.columns if c not in ("match_id", "season")]
    return df[cols]


def validate(df: pd.DataFrame) -> None:
    """Fail loudly if the data doesn't look like 12 full Premier League seasons."""
    assert df["date"].notna().all(), "Unparsed dates"
    assert not df.duplicated(["date", "home_team", "away_team"]).any(), "Duplicate matches"
    assert df["result"].isin(["H", "D", "A"]).all(), "Unexpected result codes"

    goal_diff = df["home_goals"] - df["away_goals"]
    expected = goal_diff.apply(lambda d: "H" if d > 0 else "A" if d < 0 else "D")
    assert (expected == df["result"]).all(), "Result doesn't match the score"

    for season, games in df.groupby("season"):
        assert len(games) == 380, f"{season}: {len(games)} matches, expected 380"
        teams = set(games["home_team"]) | set(games["away_team"])
        assert len(teams) == 20, f"{season}: {len(teams)} teams, expected 20"
        # Every team plays every other team home and away exactly once.
        pairs = games.groupby(["home_team", "away_team"]).size()
        assert len(pairs) == 380 and (pairs == 1).all(), f"{season}: fixture list is wrong"


def build(download: bool = False) -> pd.DataFrame:
    if download:
        download_seasons()
    seasons = [load_season(y) for y in range(FIRST_SEASON, LAST_SEASON + 1)]
    df = clean(pd.concat(seasons, ignore_index=True))
    validate(df)
    return df


def load_matches() -> pd.DataFrame:
    """Load the processed table (use this from notebooks)."""
    return pd.read_csv(PROCESSED_PATH, parse_dates=["date"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="download missing raw seasons")
    args = parser.parse_args()

    matches = build(download=args.download)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved {len(matches)} matches, {matches['season'].nunique()} seasons -> {PROCESSED_PATH.relative_to(ROOT)}")
