import pandas as pd
import datetime

def fill_missing(df: pd.DataFrame, exclude=('Latest_date',)) -> pd.DataFrame:
    """
    Заполняет пропуски:
      - в колонках, где 'probability' в имени (case-insensitive) -> медианой колонки
      - в остальных (кроме exclude) -> нулём
    """
    df = df.copy()
    for col in df.columns:
        if col in exclude:
            continue
        if 'probability' in col.lower():
            median_val = df[col].median()
            # если медиана NaN (например, все значения пустые) — оставляем 0 как fallback
            df[col] = df[col].fillna(0) # median_val if pd.notna(median_val) else 0)
        else:
            df[col] = df[col].fillna(0)
    return df


last = pd.read_excel('data_cur/active_group_predict_july_plus_date_2.xlsx')
new = pd.read_csv('data_cur/predictions_aug_processed.csv')

merged = last.merge(new, on='INN', how='left')
merged = fill_missing(merged)

merged.to_excel('data_cur/active_group_predict_aug.xlsx')


t1 = pd.read_csv('data/v21/trn1.csv')
t2 = pd.read_csv('data/v21/trn2.csv')
t3 = pd.read_csv('data/v21/trn3.csv')
tt = pd.read_csv('data/v21/test_april2026.csv')

t = pd.concat([t1, t2, t3, tt])[['INN', 'Latest_date']]
t['Latest_date'] = pd.to_datetime(t['Latest_date'])
t = t.sort_values('Latest_date', ascending=False).drop_duplicates('INN', keep='first')
print(t.columns)

t = merged.merge(t, how='left', on='INN')
t = fill_missing(t)

print(t.columns)

t = t[pd.to_datetime(t['Latest_date']).dt.date >= datetime.date(year=2025, month=9, day=1)]
t.to_excel('data_cur/active_group_predict_aug_plus_date_2.xlsx')