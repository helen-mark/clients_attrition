import pandas as pd
import numpy as np

# Загружаем данные
predictions_df = pd.read_excel('data_cur/predictions_aug.xlsx')

# Загружаем группы
active_df = pd.read_csv('new_active_final.csv')
control_df = pd.read_csv('new_control_final.csv')
placebo_df = pd.read_csv('new_placebo_final.csv')

print(f"Исходный размер predictions: {len(predictions_df)}")
print(f"Размер active группы: {len(active_df)}")
print(f"Размер control группы: {len(control_df)}")
print(f"Размер placebo группы: {len(placebo_df)}")

# 1. Удаляем control
predictions_df = predictions_df[~predictions_df['INN'].isin(control_df['INN'])]
print(f"\nПосле удаления control группы: {len(predictions_df)}")

# --- Определяем пары (probability_col, prediction_col) по суффиксу ---
prob_cols = [c for c in predictions_df.columns if 'probabilit' in c.lower()]
print(f"\nНайдены probability-колонки: {prob_cols}")

# Строим пары: Probabilities_<suffix> ↔ Prediction_<suffix>
pairs = []
for prob_col in prob_cols:
    # Извлекаем суффикс после '_' (например, 'aug' из 'Probabilities_aug')
    suffix = prob_col.split('_', 1)[1] if '_' in prob_col else ''
    # Ищем соответствующий Prediction-колонку
    pred_col_candidates = [c for c in predictions_df.columns
                           if c.lower().startswith('prediction')
                           and c.split('_', 1)[1] == suffix]
    if pred_col_candidates:
        pred_col = pred_col_candidates[0]
        pairs.append((prob_col, pred_col, suffix))
        print(f"  Пара: {prob_col} <-> {pred_col}")
    else:
        print(f"  ВНИМАНИЕ: не найдена Prediction-колонка для {prob_col} (суффикс '{suffix}')")

if not pairs:
    raise ValueError("Не найдено ни одной пары probability<->prediction. Проверьте названия колонок.")

# Маски
placebo_mask = predictions_df['INN'].isin(placebo_df['INN'])
active_mask = predictions_df['INN'].isin(active_df['INN'])

n_placebo_rows = placebo_mask.sum()
n_active_rows = active_mask.sum()
print(f"\nСтрок placebo: {n_placebo_rows}, строк active: {n_active_rows}")

np.random.seed(42)

TARGET_POSITIVE_RATE = 0.13  # 13%

# Индексы строк placebo и active
placebo_index = predictions_df.index[placebo_mask]
active_index = predictions_df.index[active_mask]

# --- Обрабатываем каждую пару независимо ---
for prob_col, pred_col, suffix in pairs:
    print(f"\n=== Обработка пары: {prob_col} <-> {pred_col} (суффикс '{suffix}') ===")

    # Пул значений probability из active
    active_probs = predictions_df.loc[active_index, prob_col]

    # Порог по active: квантиль (1 - 0.13) = 0.87
    threshold = np.quantile(active_probs, 1 - TARGET_POSITIVE_RATE)
    print(f"  Порог (квантиль {1 - TARGET_POSITIVE_RATE:.2f}) по active: {threshold:.6f}")
    predictions_df.loc[active_index, pred_col] = (
            predictions_df.loc[active_index, prob_col] >= threshold
    ).astype(int)

    # Доля положительных в active при этом пороге
    active_pos_rate = (active_probs >= threshold).mean()
    print(f"  Доля положительных в active при пороге: {active_pos_rate:.4f}")

    # Заимствуем probability из active для placebo (строки целиком, чтобы сохранить совместное распределение,
    # но здесь обрабатываем пары независимо — если нужно сохранить корреляцию между разными месяцами,
    # см. вариант ниже)
    sampled_idx = np.random.choice(active_index, size=n_placebo_rows, replace=True)
    sampled_probs = predictions_df.loc[sampled_idx, prob_col].values

    # Присваиваем placebo
    predictions_df.loc[placebo_index, prob_col] = sampled_probs

    # Пересчитываем соответствующий Prediction по общему порогу
    predictions_df.loc[placebo_index, pred_col] = (
        predictions_df.loc[placebo_index, prob_col] >= threshold
    ).astype(int)

    # Диагностика по placebo
    placebo_probs = predictions_df.loc[placebo_index, prob_col]
    placebo_pos_rate = (placebo_probs >= threshold).mean()
    print(f"  Доля положительных в placebo при пороге: {placebo_pos_rate:.4f}")

    # Проверка на уровне уникальных ИНН
    placebo_inn_pos = (
        predictions_df.loc[placebo_index]
        .groupby('INN')[pred_col].max().sum()
    )
    placebo_inn_total = predictions_df.loc[placebo_index, 'INN'].nunique()
    print(f"  Placebo: уникальных ИНН {placebo_inn_total}, "
          f"с положительным {pred_col}: {placebo_inn_pos} "
          f"({placebo_inn_pos / placebo_inn_total * 100:.2f}%)")

# --- Если нужно сохранить корреляцию между probability-колонками разных месяцев ---
# Тогда вместо независимой обработки каждой пары, заимствуйте строки целиком:
#
# all_prob_cols = [p[0] for p in pairs]
# sampled_idx = np.random.choice(active_index, size=n_placebo_rows, replace=True)
# sampled_block = predictions_df.loc[sampled_idx, all_prob_cols].reset_index(drop=True)
# for prob_col in all_prob_cols:
#     predictions_df.loc[placebo_index, prob_col] = sampled_block[prob_col].values
# Затем для каждой пары пересчитайте Prediction по своему порогу.

# --- Проверка control ---
control_in_predictions = predictions_df['INN'].isin(control_df['INN']).sum()
print(f"\nСтрок control группы в результате: {control_in_predictions} (должно быть 0)")

# --- Сохраняем ---
predictions_df.to_csv('data_cur/predictions_aug_processed.csv', index=False)
print(f"\nРезультат сохранен в csv")
print(f"Финальный размер файла: {len(predictions_df)} строк")