from __future__ import annotations

import os
import pandas as pd
from dateutil.relativedelta import relativedelta

# Папка с CSV-файлами
FOLDER = "data/v26"

# --- Условие 1: горизонт "видимости" в месяцах ---
CHURN_HORIZON_MONTHS = 6

# --- Условие 2: статичная дата-отсечка ---
# Если Latest_date > STATIC_CUTOFF_DATE → принудительно 0.
# None — отключить.
STATIC_CUTOFF_DATE = "2026-03-01"  # to cutoff test data to avoid leakage. None if not used

# --- Условие 3: пост-условие по сезонности ---
# Если Latest_date >= SEASONALITY_START, то обнуляем, когда:
#   Seasonality in (1, 2)  ИЛИ  Latest_date >= HARD_ZERO_DATE
SEASONALITY_START = "2026-03-01"  # maybe just left for summer
HARD_ZERO_DATE    = "2026-08-01"  # Put CURRENT date here because if Latest_date is now then the client is active
SEASONALITY_VALUES = (1, 2)

# Колонки
COL_TARGET     = "ACTIVITY_AND_ATTRITION"
COL_LATEST     = "Latest_date"
COL_BOUND      = "upper_bound"
COL_SEASONALITY = "Seasonality"


def recompute_activity(
    df: pd.DataFrame,
    horizon_months: int = CHURN_HORIZON_MONTHS,
    static_cutoff: str | None = STATIC_CUTOFF_DATE,
    seasonality_start: str | None = SEASONALITY_START,
    hard_zero_date: str | None = HARD_ZERO_DATE,
    seasonality_values: tuple = SEASONALITY_VALUES,
) -> pd.DataFrame:
    latest = pd.to_datetime(df[COL_LATEST], errors="coerce")
    bound  = pd.to_datetime(df[COL_BOUND], errors="coerce")

    # --- Условие 1: базовый churn ---
    threshold = bound.apply(
        lambda d: d + relativedelta(months=horizon_months) if pd.notna(d) else pd.NaT
    )
    churn = (latest.notna() & threshold.notna() & (latest <= threshold)).astype(int)

    # --- Условие 2: статичная отсечка (перекрывает 1) ---
    if static_cutoff is not None:
        cutoff = pd.to_datetime(static_cutoff)
        churn = churn.mask(latest.notna() & (latest > cutoff), 0)

    # --- Условие 3: пост-условие по сезонности (перекрывает всё) ---
    if seasonality_start is not None and COL_SEASONALITY in df.columns:
        season_start = pd.to_datetime(seasonality_start)
        seasonality = pd.to_numeric(df[COL_SEASONALITY], errors="coerce")

        mask_ge_start = latest.notna() & (latest >= season_start)

        cond_season = mask_ge_start & seasonality.isin(seasonality_values)

        if hard_zero_date is not None:
            hard = pd.to_datetime(hard_zero_date)
            cond_hard = latest.notna() & (latest >= hard)
        else:
            cond_hard = pd.Series(False, index=df.index)

        churn = churn.mask(cond_season | cond_hard, 0)

    df = df.copy()
    df[COL_TARGET] = churn
    return df


def process_folder(folder: str) -> None:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Папка не найдена: {folder}")

    for name in os.listdir(folder):
        if not name.lower().endswith(".csv"):
            continue

        if '000' in name:
            continue

        path = os.path.join(folder, name)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[SKIP] {name}: не удалось прочитать ({e})")
            continue

        required = [COL_TARGET, COL_LATEST, COL_BOUND]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"[SKIP] {name}: отсутствуют колонки {missing}")
            continue

        if COL_SEASONALITY not in df.columns:
            print(f"[WARN] {name}: нет колонки {COL_SEASONALITY}, условие 3 частично пропущено")

        df = recompute_activity(df)

        df.to_csv(path, index=False)
        n_churn = int(df[COL_TARGET].sum())
        print(f"[OK]   {name}: обработано {len(df)} строк, ушедших = {n_churn}")


if __name__ == "__main__":
    process_folder(FOLDER)