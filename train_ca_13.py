"""
  Created on Jan 2025
@author: Elena Markova
          for Attrition Rate Project
"""

import os

os.environ['YDATA_LICENSE_KEY'] = '97d0ae93-9dfc-4c2a-9183-a0420a4d0771'

import pickle
import warnings
from scipy import stats

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from optuna.pruners import MedianPruner
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score
from sklearn.metrics import f1_score, precision_recall_curve, roc_curve

from sklearn.model_selection import train_test_split, cross_val_score
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
from catboost import CatBoostClassifier, Pool
from sklearn.feature_selection import RFECV

import shap

from pathlib import Path
# from examples.local import setting_dask_env

import seaborn as sn
import matplotlib.pyplot as plt

from test_ca import test
from utils.dataset_13 import create_features_for_datasets, collect_datasets, minority_class_resample, prepare_dataset_2
from eval_model import test_rowwise
# industry_avg_income = df.groupby('field')['income_shortterm'].mean().to_dict()
# df['industry_avg_income'] = df['field'].map(industry_avg_income)
# df['income_vs_industry'] = df['income_shortterm'] - df['industry_avg_income']
# position_median_income = df.groupby('department')['income_shortterm'].median().to_dict()
# df['position_median_income'] = df['department'].map(position_median_income)

def show_importance(_x_train, _y_train, _model):
    train_pool = Pool(data=_x_train, label=_y_train)
    shap_values = _model.get_feature_importance(prettified=False, type='ShapValues', data=train_pool)
    feature_names = _x_train.columns
    #base_value = shap_values[0, -1]  # Последний столбец для всех samples одинаков
    #print(f"Base value (средняя вероятность класса 1): {base_value:.4f}")
    importance = ((shap_values[:, :-1])).mean(axis=0)
    importance_abs = (abs(shap_values[:, :-1])).mean(axis=0)

    sorted_idx = np.argsort(np.abs(importance))  # Indices from highest to lowest magnitude
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_shap = [importance[i] for i in sorted_idx]  # Signed values
    for f, i in zip(sorted_features, sorted_shap):
        #print(f"{f}: {i}")
        if i > 0.000001:
            print(f)
    sorted_abs_shap = [importance_abs[i] for i in sorted_idx]  # Unsigned values
    colors = ['red' if val > 0 else 'blue' for val in sorted_shap]
    sorted_shap = [abs(s) for s in sorted_shap]  # Signed values
    plt.barh(sorted_features[-35:], sorted_shap[-35:], color=colors)
    plt.title('CatBoost Feature Importance')
    plt.show()
    plt.clf()
    plt.barh(sorted_features[-35:], sorted_abs_shap[-35:])
    plt.title('Catboost Feature Importance (unsigned absolute values)')
    plt.show()

    explainer = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(_x_train)

    # График зависимости SHAP от значения фичи
    # shap.dependence_plot("weather_winter_sum_0", shap_values, _x_train)
    # shap.dependence_plot("weather_sum_0", shap_values, _x_train)
    # shap.dependence_plot("weather_sum_1", shap_values, _x_train)

    #shap.dependence_plot("MEDIAN_PRICE", shap_values, _x_train)
    # shap.dependence_plot("AVG_PRICE_high", shap_values, _x_train)
    # shap.dependence_plot("AVG_PRICE_medium", shap_values, _x_train)
    # shap.dependence_plot("AVG_PRICE_low", shap_values, _x_train)
    # shap.dependence_plot("AVG_PRICE_medium_high", shap_values, _x_train)
    # shap.dependence_plot("AVG_PRICE_medium_low", shap_values, _x_train)

    #shap.dependence_plot("Firm_age_months", shap_values, _x_train)
    #shap.dependence_plot("drivers_per_address", shap_values, _x_train)
    #shap.dependence_plot("РентАктивов_Много ниже нормы", shap_values, _x_train)
    #shap.dependence_plot("РентАктивов_Выше нормы", shap_values, _x_train)
    #shap.dependence_plot("_bankrots2016БаллЗона_deriv", shap_values, _x_train)
    #shap.dependence_plot("_bankrots2016БаллЗона_after_Высокий балл", shap_values, _x_train)
    #shap.dependence_plot("_bankrots2016БаллЗона_after_Средний балл", shap_values, _x_train)
    #shap.dependence_plot("_bankrots2016БаллЗона_after_Низкий балл", shap_values, _x_train)
    #shap.dependence_plot("_problemCredit_БаллЗона_deriv", shap_values, _x_train)
    #shap.dependence_plot("_problemCredit_БаллЗона_after_Высокий балл", shap_values, _x_train)
    #shap.dependence_plot("_problemCredit_БаллЗона_after_Средний балл", shap_values, _x_train)
    #shap.dependence_plot("_problemCredit_БаллЗона_after_Низкий балл", shap_values, _x_train)


def train_catboost(_x_train, _y_train, _x_test, _y_test, _sample_weight, _cat_feats_encoded, _num_iters):
    # model already initialized with latest version of optimized parameters for our dataset

    model = CatBoostClassifier(
        iterations=562, #400  # Fewer trees + early stopping
        learning_rate=0.075,  #0.08,  # Smaller steps for better generalization
        depth=6,
        l2_leaf_reg=5, #10,  # Stronger L2 regularization
        bootstrap_type='MVS',
        # bagging_temperature=1,  # Less aggressive subsampling
        random_strength=1,  # Default randomness
        loss_function='Logloss',
        eval_metric='AUC',
        auto_class_weights='Balanced',  # Adjust for class imbalance
        od_type='Iter',  # Early stopping
        od_wait=51,  # Patience before stopping
    )
    #CatBoostClassifier._clear_training_state()  # Critical for looped execution

    # model = CatBoostClassifier(
    #     iterations=543,  # 400  # Fewer trees + early stopping
    #     learning_rate=0.12,  # 0.08,  # Smaller steps for better generalization
    #     depth=4,  # Slightly deeper but not excessive
    #     l2_leaf_reg=9,  # 10,  # Stronger L2 regularization
    #     bootstrap_type='MVS',
    #     # bagging_temperature=1,  # Less aggressive subsampling
    #     random_strength=2,  # Default randomness
    #     loss_function='Logloss',
    #     eval_metric='AUC',
    #     auto_class_weights='Balanced',
    #     # auto_class_weights='Balanced',  # Adjust for class imbalance
    #     od_type='IncToDec',  # Early stopping
    #     od_wait=65,  # Patience before stopping
    #     silent=True
    # )

    selector = RFECV(
        estimator=model,
        step=1,
        cv=3,
        scoring='roc_auc'
    )
    #selector.fit(_x_train, _y_train)

    #selected_features = _x_train.columns[selector.support_]
    #print(f'Selected features: {selected_features}')

    # model = CatBoostClassifier(
    #     iterations=933, #400  # Fewer trees + early stopping
    #     learning_rate=0.17,  #0.08,  # Smaller steps for better generalization
    #     depth=6,  # Slightly deeper but not excessive
    #     l2_leaf_reg=7, #10,  # Stronger L2 regularization
    #     bootstrap_type='MVS',
    #     # bagging_temperature=1,  # Less aggressive subsampling
    #     random_strength=2,  # Default randomness
    #     loss_function='Logloss',
    #     eval_metric='F1',
    #     # auto_class_weights='Balanced',  # Adjust for class imbalance
    #     od_type='IncToDec',  # Early stopping
    #     od_wait=47,  # Patience before stopping
    #     random_seed=42
    # )


    def objective(trial):
        train_pool = Pool(_x_train, _y_train)
        eval_pool = Pool(_x_test, _y_test)
        params={'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True), 'iterations': trial.suggest_int('iterations', 100, 1000), 'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_int('l2_leaf_reg1', 3, 10),
                'od_wait': trial.suggest_int('od_wait', 20, 80), 'od_type': trial.suggest_categorical('od_type', ['Iter', 'IncToDec']),
                #'bagging_temperature': trial.suggest_float('bagging_temperature', 0.5, 1),
                'random_strength': trial.suggest_int('random_strength', 1, 3),
                'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'MVS'])}
        model = CatBoostClassifier(**params, verbose=0)
        score = cross_val_score(model, pd.concat([_x_train, _x_test]), pd.concat([_y_train, _y_test]), cv=3, scoring='roc_auc').mean()
        return score
    #
    # study = optuna.create_study(direction='maximize')
    # study.optimize(objective, n_trials=370)
    # print(f"Best parameters of optuna: {study.best_params}")

    model.fit(
        _x_train,
        _y_train,
        eval_set=(_x_test, _y_test),
        verbose=False,
        # sample_weight=_sample_weight,
        # plot=True,
        # cat_features=_cat_feats_encoded - do this if haven't encoded cat features
    )
    # model = pickle.load(open('model_Rec_70_prec_40_thres_05.pkl', 'rb'))
    show_importance(_x_train, _y_train, model)

    return model


def train(_x_train, _y_train, _x_test, _y_test, _sample_weight, _cat_feats_encoded, _model_name, _num_iters):
    if _model_name == "CatBoostClassifier":
        model = train_catboost(_x_train, _y_train, _x_test, _y_test, _sample_weight, _cat_feats_encoded, _num_iters)
    else:
        print("Model name error: this model is not implemented yet!")
        return

    return model


def prepare_dataset(_dataset: pd.DataFrame, _test_split: float, _normalize: bool):
    target_idx = -1  # index of "works/left" column

    dataset = _dataset.transpose()
    trg = dataset[target_idx:]
    trn = dataset[:target_idx]

    # val_size = 2000
    # trn = trn.transpose()
    # trg = trg.transpose()
    # x_train = trn[val_size:]
    # x_test = trn[:val_size]
    # y_train = trg[val_size:]
    # y_test = trg[:val_size]

    x_train, x_test, y_train, y_test = train_test_split(trn.transpose(), trg.transpose(), test_size=_test_split)

    if _normalize:  # normalization is NOT needed for decision trees!
        x_train = normalize(x_train)
        x_test  = normalize(x_test)

    return x_train, x_test, y_train, y_test


def test_with_slices(_model, trn, trg, target_positive_rates=[0.03, 0.1, 0.2]):
    """
    Тестирование модели на всем датасете и на срезах предсказанных положительных объектов
    """
    y_proba = _model.predict_proba(trn)[:, 1]
    precision, recall, thresholds = precision_recall_curve(trg, y_proba)

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]

    # Calculate thresholds for target positive rates
    sorted_proba = np.sort(y_proba)[::-1]
    percentile_thresholds = {}

    for target_rate in target_positive_rates:
        n_positive = int(len(y_proba) * target_rate)
        if n_positive > 0 and n_positive <= len(sorted_proba):
            percentile_threshold = sorted_proba[n_positive - 1]
            percentile_thresholds[target_rate] = percentile_threshold
            print(f"Threshold for {target_rate * 100:.0f}% positive rate: {percentile_threshold:.4f}")

    # Combine all thresholds
    all_thresholds = [optimal_threshold] + list(percentile_thresholds.values())
    threshold_labels = ['Optimal F1'] + [f'{rate * 100:.0f}% Positive' for rate in target_positive_rates]

    N_1 = len([y for y in trg if y == 1])
    N_0 = len([y for y in trg if y == 0])

    def adjusted_precision(P, N, M, new_N, new_M):
        numerator = P * (new_N / N)
        denominator = numerator + (1 - P) * (new_M / M)
        return numerator / denominator

    results = []

    for i, thrs in enumerate(all_thresholds):
        print(f"\n{'=' * 60}")
        print(f"Testing with threshold = {thrs:.4f} ({threshold_labels[i]})")
        print(f"{'=' * 60}")

        predictions = _model.predict_proba(trn)
        predicted_proba = predictions[:, 1]
        predictions = (predicted_proba > thrs).astype(int)

        actual_positive_rate = predictions.mean()

        # Метрики на всем датасете
        f1_united = f1_score(trg, predictions)
        recall_united = recall_score(trg, predictions)
        precision_united = precision_score(trg, predictions)
        precision_united_adjusted = adjusted_precision(precision_united, N_1, N_0, 1000, 9000)
        a = balanced_accuracy_score(trg, predictions)

        print(
            f"CatBoost result: F1 = {f1_united:.2f}, Recall = {recall_united:.2f}, Precision = {precision_united:.2f}")
        print(f"Adjusted Precision = {precision_united_adjusted:.2f}, Balanced Accuracy = {a:.2f}")
        print(f"Actual Positive Rate: {actual_positive_rate * 100:.1f}%")

        # НОВОЕ: Метрики на срезе предсказанных положительных
        predicted_positive_mask = predictions == 1
        n_predicted_positive = predicted_positive_mask.sum()

        if n_predicted_positive > 0:
            # Срез данных - только предсказанные положительные
            trg_slice = trg[predicted_positive_mask]
            proba_slice = predicted_proba[predicted_positive_mask]

            # Количество TP и FP в срезе
            n_true_positive_in_slice = (trg_slice == 1).sum()
            n_false_positive_in_slice = (trg_slice == 0).sum()

            # Precision в срезе (это то же самое, что и precision_united)
            precision_in_slice = n_true_positive_in_slice / n_predicted_positive

            # Adjusted precision для среза (пересчет на новые пропорции классов)
            precision_slice_adjusted = adjusted_precision(
                precision_in_slice,
                n_true_positive_in_slice,
                n_false_positive_in_slice,
                1000,  # новое ожидаемое количество positives
                9000  # новое ожидаемое количество negatives
            )

            # Средняя вероятность в срезе
            mean_proba_in_slice = proba_slice.mean()
            median_proba_in_slice = np.median(proba_slice)

            # Распределение вероятностей в срезе
            proba_std_in_slice = proba_slice.std()

            print(f"\n--- Metrics for predicted positive slice ---")
            print(f"Number of predicted positives: {n_predicted_positive}")
            print(f"True positives in slice: {n_true_positive_in_slice}")
            print(f"False positives in slice: {n_false_positive_in_slice}")
            print(f"Precision in slice: {precision_in_slice:.3f}")
            print(f"Adjusted precision in slice: {precision_slice_adjusted:.3f}")
            print(f"Mean probability in slice: {mean_proba_in_slice:.3f}")
            print(f"Median probability in slice: {median_proba_in_slice:.3f}")
            print(f"Std probability in slice: {proba_std_in_slice:.3f}")

            # Если есть истинные положительные в срезе
            if n_true_positive_in_slice > 0:
                # Recall относительно всех истинных положительных
                recall_of_all_positives = n_true_positive_in_slice / N_1
                print(f"Recall of all positives (coverage): {recall_of_all_positives:.3f}")

            # Визуализация распределения вероятностей в срезе
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Распределение вероятностей для TP и FP в срезе
            tp_proba = proba_slice[trg_slice == 1]
            fp_proba = proba_slice[trg_slice == 0]

            axes[0].hist(tp_proba, bins=30, color='green', alpha=0.6, label=f'TP (n={len(tp_proba)})',
                         edgecolor='black')
            axes[0].hist(fp_proba, bins=30, color='red', alpha=0.6, label=f'FP (n={len(fp_proba)})', edgecolor='black')
            axes[0].axvline(x=thrs, color='blue', linestyle='--', linewidth=2, label=f'Threshold = {thrs:.3f}')
            axes[0].set_xlabel('Predicted Probability')
            axes[0].set_ylabel('Count')
            axes[0].set_title(f'Probability Distribution in Predicted Positive Slice')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Box plot для сравнения распределений
            data_to_plot = [tp_proba, fp_proba]
            bp = axes[1].boxplot(data_to_plot, labels=['True Positives', 'False Positives'], patch_artist=True)
            bp['boxes'][0].set_facecolor('green')
            bp['boxes'][1].set_facecolor('red')
            axes[1].axhline(y=thrs, color='blue', linestyle='--', linewidth=2, label=f'Threshold = {thrs:.3f}')
            axes[1].set_ylabel('Predicted Probability')
            axes[1].set_title(f'Box Plot: TP vs FP in Slice')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.suptitle(f'Analysis of Predicted Positive Slice - {threshold_labels[i]}', fontsize=14)
            plt.tight_layout()
            plt.show()

            # Сохраняем результаты для среза
            slice_result = {
                'threshold_label': threshold_labels[i],
                'threshold': thrs,
                'slice_size': n_predicted_positive,
                'slice_precision': precision_in_slice,
                'slice_adjusted_precision': precision_slice_adjusted,
                'slice_mean_proba': mean_proba_in_slice,
                'slice_median_proba': median_proba_in_slice,
                'slice_std_proba': proba_std_in_slice,
                'slice_tp': n_true_positive_in_slice,
                'slice_fp': n_false_positive_in_slice,
                'coverage_of_all_positives': n_true_positive_in_slice / N_1 if N_1 > 0 else 0
            }
        else:
            print(f"\nNo predicted positives for this threshold!")
            slice_result = {
                'threshold_label': threshold_labels[i],
                'threshold': thrs,
                'slice_size': 0,
                'slice_precision': 0,
                'slice_adjusted_precision': 0,
                'slice_mean_proba': 0,
                'slice_median_proba': 0,
                'slice_std_proba': 0,
                'slice_tp': 0,
                'slice_fp': 0,
                'coverage_of_all_positives': 0
            }

        results.append({
            'threshold_label': threshold_labels[i],
            'threshold': thrs,
            'f1': f1_united,
            'recall': recall_united,
            'precision': precision_united,
            'adjusted_precision': precision_united_adjusted,
            'balanced_accuracy': a,
            'actual_positive_rate': actual_positive_rate,
            'slice_metrics': slice_result
        })

    # Print summary comparison
    print(f"\n{'=' * 80}")
    print("SUMMARY COMPARISON")
    print(f"{'=' * 80}")
    print(
        f"{'Threshold Type':<20} {'Threshold':<10} {'F1':<8} {'Precision':<10} {'Slice Prec':<12} {'Slice Adj Prec':<14} {'Coverage':<10}")
    print("-" * 84)
    for res in results:
        slice_m = res['slice_metrics']
        print(f"{res['threshold_label']:<20} {res['threshold']:<10.4f} {res['f1']:<8.3f} {res['precision']:<10.3f} "
              f"{slice_m['slice_precision']:<12.3f} {slice_m['slice_adjusted_precision']:<14.3f} "
              f"{slice_m['coverage_of_all_positives'] * 100:<10.1f}%")

    return results



def select_threshold(y_true, y_proba):
    if len(np.unique(y_true)) != 2:
        raise ValueError('Validation must contain both classes to select a balanced-accuracy threshold')
    false_positive, true_positive, thresholds = roc_curve(y_true, y_proba)
    finite = np.isfinite(thresholds)
    return float(thresholds[finite][np.argmax((true_positive - false_positive)[finite])])


def remove_test_duplicates(x_train, y_train, x_test):
    duplicates = pd.MultiIndex.from_frame(x_train).isin(pd.MultiIndex.from_frame(x_test[x_train.columns]))
    print(f'Number of duplicate training rows: {int(duplicates.sum())}')
    return (x_train.iloc[~duplicates].reset_index(drop=True),
            y_train.iloc[~duplicates].reset_index(drop=True))


# Example usage:
# results = test(model, X_test, y_test, target_positive_rates=[0.05, 0.1, 0.15, 0.2])

def calc_weights(_y_train: pd.DataFrame, _y_val: pd.DataFrame):
    w_0, w_1 = 0, 0
    for i in _y_train.values:
        w_0 += 1 - i.item()
        w_1 += i.item()

    tot = w_0 + w_1
    w_0 = w_0 / tot
    w_1 = w_1 / tot
    print(f"weights:", w_0, w_1)
    #return np.array([w_0 if i == 1 else w_1 for i in _y_train.values])
    return np.array([1 if i == 1 else 1 for i in _y_train.values])

def remove_outliers(dfs):
    # Features to process
    features_to_trim = [
        c for c in dfs[0].columns if
        'AVG_PRICE' in c or
        'MEDIAN_PRICE' in c or
        'Price' in c or
        'Turnover' in c or
        'Dur_months' in c
    ]
    if not any(feature in dfs[0].columns for feature in features_to_trim):
        return dfs
    new_datasets = []
    for df in dfs:
        for f in features_to_trim:
            if f not in df.columns:
                continue
            f = [f]
        # Step 1: Drop rows with NaN in features_to_trim (critical!)
            df = df.dropna(subset=f).copy()

        # Step 2: Skip features with zero variance (avoid NaN z-scores)
        # valid_features = [f for f in features_to_trim if df[f].var() > 0]
        # if not valid_features:
        #     new_datasets.append(df)  # No features left to filter
        #     continue

        # Step 3: Calculate Z-scores only on valid features
            z_scores = np.abs(stats.zscore(df[f]))
            filtered_rows = (z_scores < 2.5).all(axis=1)
            df_clean = df[filtered_rows].copy().reset_index(drop=True)  # Reset index here


        # # Step 4: Verify no NaN in target
        # assert df_clean['status'].isna().sum() == 0, \
        #     f"Target has {df_clean['status'].isna().sum()} NaN values after filtering!"

        new_datasets.append(df_clean)

        # Debug info
        print(f"Original rows: {len(df)}")
        print(f"Rows after outlier removal: {len(df_clean)}")
        print(f"Rows removed: {len(df) - len(df_clean)}")
    return new_datasets


import pandas as pd


def filter_by_attrition_reason(df):
    print(df['Reason'].unique())

def remove_non_addressable(df):
    print('-'*50)
    print("Before removing non-addressable:", len(df[df['ACTIVITY_AND_ATTRITION']==1]))
    addressable_df = pd.read_excel('reports/Причины оттока.xlsx')  #('data_cur/mapping/merged.csv')

    def normalize_inn(series):
        # приводим к строке, убираем .0, пробелы и т.п.
        return (
            series
            .astype(str)
            .str.strip()
            .str.replace(r'\.0$', '', regex=True)
        )

    addressable_inns = set(
        normalize_inn(addressable_df.loc[addressable_df['Addressable'] == 'Non actionable', 'INN'])
    )

    mask = normalize_inn(df['INN']).isin(addressable_inns)

    #print(f"Найдено неподходящих ИНН: {mask.sum()} из {len(df)}")

    df.loc[mask, 'ACTIVITY_AND_ATTRITION'] = 0
    print("After removing non-addressable:", len(df[df['ACTIVITY_AND_ATTRITION']==1]))

    return df

def calculate_mail_metrics(df, metrics_file='data_cur/inn_metrics.csv'):
    """
    Adds metric_1 and metric_2 columns to the input dataframe based on INN.
    Removes rows where INN is not found in the metrics file.

    Parameters:
    df (pd.DataFrame): Input dataframe with 'INN' column
    metrics_file (str): Path to the CSV file with metrics

    Returns:
    pd.DataFrame: Dataframe with added metric columns, filtered to only matching INNs
    """
    # Load metrics file
    df_metrics = pd.read_csv(metrics_file)

    # Merge with input dataframe on INN (inner join removes non-matching INNs)
    df_result = df.merge(df_metrics, on='INN', how='inner')

    # Report how many rows were removed
    rows_before = len(df)
    rows_after = len(df_result)
    rows_removed = rows_before - rows_after

    if rows_removed > 0:
        print(
            f"Removed {rows_removed} rows ({(rows_removed / rows_before) * 100:.1f}%) - INNs not found in metrics file")
    else:
        print(f"All {rows_before} INNs found in metrics file")

    return df_result


# Example usage:
# df = pd.read_csv('some_file.csv')
# df_with_metrics = add_metrics(df)
# print(df_with_metrics.head())

# Alternative usage with different metrics file:
# df_with_metrics = add_metrics(df, 'path/to/custom_metrics.csv')

def main(_config: dict):
    if 'datasets' in globals():
        del globals()['datasets']

    data_path = _config['dataset_src']
    datasets = collect_datasets(data_path)
    if len(datasets) < 3:
        raise ValueError('At least three snapshots are required: test, training and validation')
    rand_states = [4]
    score = [0, 0, 0]

    if _config['filter_by_attrition_reason']:
        datasets = [filter_by_attrition_reason(df) for df in datasets]

    if _config['remove_non_addressable']:
        datasets = [remove_non_addressable(df) for df in datasets]

    if _config['use_mail_metrics']:
        datasets = [calculate_mail_metrics(df) for df in datasets]

    if _config['remove_outliers']:
        datasets = [df.copy().reset_index(drop=True) for df in datasets]
        datasets = remove_outliers(datasets)

    if _config['calculated_features']:
        datasets, new_cat_feat = create_features_for_datasets(datasets, _config, _validation_index=len(datasets) - 1)
        _config['cat_features'] += new_cat_feat

    for split_rand_state in rand_states:
        d_train, d_validation, d_test, cat_feats_encoded = prepare_dataset_2(
            datasets[:-1], _config['normalize'], _config['make_synthetic'],
            _config['encode_categorical'], _config['cat_features'],
            split_rand_state, _validation_dataset=datasets[-1]
        )
        d_test  = d_test.drop(columns=['INN'], errors='ignore')
        d_train = d_train.drop(columns=['INN'], errors='ignore')
        d_validation = d_validation.drop(columns=['INN'], errors='ignore')

        print(f"Train: {len(d_train)}, validation: {len(d_validation)}, test: {len(d_test)}")

        if _config['smote']:
            d_train = minority_class_resample(d_train, cat_feats_encoded)

        min_dur = -100
        d_train = d_train[d_train['Dur_months'] >= min_dur].copy()
        d_test  = d_test[d_test['Dur_months'] >= min_dur].copy()
        d_validation = d_validation[d_validation['Dur_months'] >= min_dur].copy()
        d_train = d_train.drop(columns=['Dur_months'], errors='ignore')
        d_test = d_test.drop(columns=['Dur_months'], errors='ignore')
        d_validation = d_validation.drop(columns=['Dur_months'], errors='ignore')


        x_train = d_train.drop('ACTIVITY_AND_ATTRITION', axis=1)
        y_train = d_train['ACTIVITY_AND_ATTRITION']
        x_val   = d_validation.drop('ACTIVITY_AND_ATTRITION', axis=1)
        y_val   = d_validation['ACTIVITY_AND_ATTRITION']
        x_test = d_test.drop('ACTIVITY_AND_ATTRITION', axis=1)
        y_test = d_test['ACTIVITY_AND_ATTRITION']

        print(f"X train: {x_train.shape[0]}, x_val: {x_val.shape[0]}, "
              f"y_train: {y_train.shape[0]}, y_val: {y_val.shape[0]}")

        print(f"Original x_train shape: {x_train.shape}")
        x_train, y_train = remove_test_duplicates(x_train, y_train, x_test)
        x_train, y_train = remove_test_duplicates(x_train, y_train, x_val)
        x_val, y_val = remove_test_duplicates(x_val, y_val, x_test)
        if x_train.empty or x_val.empty or x_test.empty:
            raise ValueError('Training, validation and test must remain nonempty after filtering')
        print(f"Cleaned x_train shape: {x_train.shape}")
        sample_weight = calc_weights(y_train, y_val)

        trained_model = train(x_train, y_train, x_val, y_val, sample_weight,
                              cat_feats_encoded, _config['model'], _config['num_iters'])

        # --- Калибровка на валидации ---
        val_proba = trained_model.predict_proba(x_val)[:, 1]
        raw_mean  = val_proba.mean()
        base_rate = (y_val == 1).mean()
        scale     = base_rate / raw_mean if raw_mean > 0 else 1.0

        print(f"Calibration scale = {scale:.4f} "
              f"(base_rate={base_rate:.4f}, raw_mean={raw_mean:.4f})")

        decision_threshold = select_threshold(y_val, np.clip(val_proba * scale, 0, 1))
        bundle = {'model': trained_model, 'calibration_scale': scale,
                  'decision_threshold': decision_threshold}

        with open('model.pkl', 'wb') as f:
            print("Saving model..")
            pickle.dump(bundle, f)

        print('Metrics on TEST set:')
        test(bundle, x_test, y_test)

    return


def train_model(config):
    """Wrapper function for training in isolated process"""
    return main(config.copy())


def save_value_to_csv(value, filename='values.csv', **metadata):
    """
    Append a floating point value to a CSV file in column 0

    Args:
        value: float value to save
        filename: name of the CSV file
    """
    # Create a DataFrame with the value
    new_row = pd.DataFrame({'values': [value], **{key: [val] for key, val in metadata.items()}})

    # Append to existing file or create new one
    if os.path.exists(filename):
        # Read existing file and append new row
        existing_df = pd.read_csv(filename)
        updated_df = pd.concat([existing_df, new_row], ignore_index=True)
    else:
        # Create new file
        updated_df = new_row

    # Save to CSV
    updated_df.to_csv(filename, index=False)
    print(f"Value {value} appended to {filename}")

if __name__ == '__main__':
    # config_path = 'config.json'  # config file is used to store some parameters of the dataset
    config = {
        'model': 'CatBoostClassifier',  # options: 'TabNet', 'RandomForestClassifier', 'XGBoostClassifier', 'CatBoostClassifier'
        'num_iters': 10,
        'normalize': False,  # normalize input values or not
        'dataset_src': 'data/v28',
        'encode_categorical': True,
        'calculated_features': True,
        'remove_outliers': False,
        'make_synthetic': None,  # options: 'sdv', 'ydata', None
        'smote': False,  # perhaps not needed for catboost and in case if minority : majority > 0.5
        'cat_features': ['Seasonality', 'legal_type'],  # , 'legal_type']  # , 'DRIVER_FIO', 'Entity Type', 'taxcode']  #, 'kbktax', 'kbknametax']  #, 'occupational_hazards']
        'use_mail_metrics': False,
        'filter_by_attrition_reason': False,
        'remove_non_addressable': False
    }

    # Use context manager for proper resource cleanup
    main(config)


# 97d0ae93-9dfc-4c2a-9183-a0420a4d0771

""" Here are some useful calculated features you could create from your existing data to enhance your analysis or model:

### **1. Client Engagement Features**
- **Average Order Value**  
  `Turnover per client per month / Number of carpet cleaning orders per client per month`  
  *(Means how much a client spends per order on average)*  

- **Carpet Cleaning Intensity**  
  `Square meters cleaned per client per month / Number of carpets per client per month`  
  *(Indicates if clients clean large carpets or many small ones)*  

- **Monthly Order Frequency**  
  `Number of carpet cleaning orders per client per month / Number of active months per year`  
  *(Shows how often clients order per active month)*  

- **Turnover per Square Meter**  
  `Turnover per client per month / Square meters cleaned per client per month`  
  *(Revenue per cleaned area – helps detect pricing differences)*  

### **2. Loyalty & Seasonality Features**
- **Client Tenure-Adjusted Activity**  
  `(Number of active months per year) / Client years with us`  
  *(Shows if long-term clients are more or less active over time)*  

- **Seasonal vs. Year-Round Ratio**  
  `(Number of active months per year) / 12`  
  *(1 = year-round, <1 = seasonal, 0.5 = winter-only, etc.)*  

- **Order Consistency Score**  
  `(Number of carpet cleaning orders per client per month) * (Client years with us)`  
  *(Higher score = loyal and consistent clients)*  

### **3. Efficiency & Business Insights**
- **Carpet Utilization Rate**  
  `Square meters cleaned per client per month / (Number of carpets × avg. carpet size)`  
  *(If you have avg. carpet size, this shows how much of their carpets they clean monthly)*  

- **Turnover per Carpet**  
  `Turnover per client per month / Number of carpets per client`  
  *(Revenue per carpet – identifies high-value clients)*  

- **Client Churn Risk Flag**  
  Binary feature: `1 if "Number of active months per year" is decreasing, else 0`  
  *(Requires historical data to detect declining activity)*  

### **4. Time-Based Features**
- **Peak Season Multiplier**  
  `(Orders in Winter) / (Orders in Summer)`  
  *(Identifies clients who heavily depend on winter cleaning)*  


Selected features: Index(['Active_months', 'Dur_months', 'Firm_age_months', 'Price',
       'SQM_SINGLE_MATS_IN_ACTIVE_SPECIFICATIONS', 'TaxPaid', 'Turnover_deriv',
       'Turnover_max_last_12', 'Turnover_median_last_12',
       'Turnover_sum_last_12', 'n_debits', 'n_drivers_per_12', 'sqm_mean',
       'sqm_median', 'sqm_sum', 'sum_debits', 'total_spacetime_area',
       'weather_sum_0', 'weather_sum_1', 'weighted_changes',
       'spacetime_area_mean', 'total_spacetime_area_fraction',
       'Turnover_mean_last_12', 'Frequency_of_changes_mean',
       'n_recalculations_mean', 'drivers_per_address'], """
