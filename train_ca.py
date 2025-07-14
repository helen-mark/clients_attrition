"""
  Created on Jan 2025
@author: Elena Markova
          for Attrition Rate Project
"""

import os

os.environ['YDATA_LICENSE_KEY'] = '97d0ae93-9dfc-4c2a-9183-a0420a4d0771'

import pickle
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from optuna.pruners import MedianPruner
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import f1_score, precision_recall_curve

from sklearn.model_selection import train_test_split, cross_val_score
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
from catboost import CatBoostClassifier, Pool

import shap

from pathlib import Path
# from examples.local import setting_dask_env

import seaborn as sn
import matplotlib.pyplot as plt

from utils.dataset_70_40 import create_features_for_datasets, collect_datasets, minority_class_resample, prepare_dataset_2, get_united_dataset

# industry_avg_income = df.groupby('field')['income_shortterm'].mean().to_dict()
# df['industry_avg_income'] = df['field'].map(industry_avg_income)
# df['income_vs_industry'] = df['income_shortterm'] - df['industry_avg_income']
# position_median_income = df.groupby('department')['income_shortterm'].median().to_dict()
# df['position_median_income'] = df['department'].map(position_median_income)


def train_xgboost_classifier(_x_train, _y_train, _x_test, _y_test, _sample_weight, _num_iters):
    model = xgb.XGBClassifier(objective="binary:logistic", random_state=42, early_stopping_rounds=15, eval_set=[(_x_test, _y_test)])
    best_f1 = 0.
    best_model = model
    test_result = {}

    print(f"Fitting XGBoost classifier...")
    for iter in range(_num_iters):
        model.fit(_x_train, _y_train, eval_set=[(_x_test, _y_test)], verbose=False)
        predictions = model.predict(_x_test)

        test_result['F1'] = f1_score(_y_test, predictions)
        if test_result['F1'] > best_f1:
            best_f1 = test_result['F1']
            best_model = model

            test_result['Recall'] = recall_score(_y_test, predictions)
            test_result['Precision'] = precision_score(_y_test, predictions)

    feature_importance = best_model.get_booster().get_score(importance_type='gain')
    keys = list(feature_importance.keys())

    # Debug print:
    values = list(feature_importance.values())
    for k, v in zip(keys, values):
        print(k)
    for k, v in zip(keys, values):
        print(v)

    data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by="score", ascending=False)
    l = data.nlargest(20, columns="score").plot(kind='barh', figsize=(20, 10))  # plot top 20 features
    print(f"XGBoost test result: Recall = {test_result['Recall']}\nPrecision = {test_result['Precision']}\nF1 = {test_result['F1']}")

    feature_importance = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': _x_train.columns, 'Importance': feature_importance})
    print(feature_importance_df)

    return best_model


def show_importance(_x_train, _y_train, _model):
    train_pool = Pool(data=_x_train, label=_y_train)
    shap_values = _model.get_feature_importance(prettified=False, type='ShapValues', data=train_pool)
    feature_names = _x_train.columns
    #base_value = shap_values[0, -1]  # Последний столбец для всех samples одинаков
    #print(f"Base value (средняя вероятность класса 1): {base_value:.4f}")
    importance = ((shap_values[:, :-1])).mean(axis=0)
    importance_abs = (abs(shap_values[:, :-1])).mean(axis=0)
    for f, i in zip(feature_names, importance):
        print(f"{f}: {i}")
    sorted_idx = np.argsort(np.abs(importance))  # Indices from highest to lowest magnitude
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_shap = [importance[i] for i in sorted_idx]  # Signed values
    sorted_abs_shap = [importance_abs[i] for i in sorted_idx]  # Unsigned values
    colors = ['red' if val > 0 else 'blue' for val in sorted_shap]
    sorted_shap = [abs(s) for s in sorted_shap]  # Signed values
    plt.barh(sorted_features, sorted_shap, color=colors)
    plt.title('CatBoost Feature Importance')
    plt.show()
    plt.clf()
    plt.barh(sorted_features, sorted_abs_shap)
    plt.title('Catboost Feature Importance (unsigned absolute values)')
    plt.show()

    explainer = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(_x_train)

    # График зависимости SHAP от значения фичи
#    shap.dependence_plot("weather_winter_sum_0", shap_values, _x_train)
#    shap.dependence_plot("weather_winter_avg_0", shap_values, _x_train)
#    shap.dependence_plot("weather_winter_sum_0", shap_values, _x_train)
#    shap.dependence_plot("weather_winter_avg_1", shap_values, _x_train)
    shap.dependence_plot("weather_sum_0", shap_values, _x_train)
    shap.dependence_plot("weather_sum_1", shap_values, _x_train)


def train_catboost(_x_train, _y_train, _x_test, _y_test, _sample_weight, _cat_feats_encoded, _num_iters):
    # model already initialized with latest version of optimized parameters for our dataset

    # v3, v7:
    # model = CatBoostClassifier(
    #     iterations=486, #400  # Fewer trees + early stopping
    #     learning_rate=0.1,  #0.08,  # Smaller steps for better generalization
    #     depth=4,  # Slightly deeper but not excessive
    #     l2_leaf_reg=4, #10,  # Stronger L2 regularization
    #     bootstrap_type='MVS',
    #     # bagging_temperature=1,  # Less aggressive subsampling
    #     random_strength=2,  # Default randomness
    #     loss_function='Logloss',
    #     eval_metric='AUC',
    #     # auto_class_weights='Balanced',  # Adjust for class imbalance
    #     od_type='IncToDec',  # Early stopping
    #     od_wait=70,  # Patience before stopping
    # )

    model = CatBoostClassifier(
        iterations=965,  # 400  # Fewer trees + early stopping
        learning_rate=0.16,  # 0.08,  # Smaller steps for better generalization
        depth=4,  # Slightly deeper but not excessive
        l2_leaf_reg=3,  # 10,  # Stronger L2 regularization
        bootstrap_type='MVS',
        # bagging_temperature=1,  # Less aggressive subsampling
        random_strength=3,  # Default randomness
        loss_function='Logloss',
        eval_metric='AUC',
        # auto_class_weights='Balanced',  # Adjust for class imbalance
        od_type='Iter',  # Early stopping
        od_wait=75,  # Patience before stopping
    )

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

    # perform parameters optimization by grid search method
    # grid = {'learning_rate': [0.03, 0.1, 0.01, 0.2], 'depth': [4,6,8,10], 'l2_leaf_reg': [1, 3, 5, 7, 10], 'od_wait': [5, 10, 20, 100], 'od_type': ['Iter', 'IncToDec'], 'bagging_temperature': [0.5, 1],
    #         'random_strength': [1, 2, 3]}
    # res = model.grid_search(grid,
    #                         X=_x_train,
    #                         y=_y_train,
    #                         # n_iter=500,
    #                         plot=True)
    #print(f"Best CatBoost params: {res}")

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
        score = cross_val_score(model, pd.concat([_x_train, _x_test]), pd.concat([_y_train, _y_test]), cv=3, scoring='f1').mean()
        return score
    #study = optuna.create_study(direction='maximize')
    #study.optimize(objective, n_trials=350)
    #print(f"Best parameters of optuna: {study.best_params}")

    model.fit(
        _x_train,
        _y_train,
        eval_set=(_x_test, _y_test),
        verbose=False,
        sample_weight=_sample_weight,
        # plot=True,
        # cat_features=_cat_feats_encoded - do this if haven't encoded cat features
    )
    # model = pickle.load(open('model_Rec_70_prec_40_thres_05.pkl', 'rb'))
    show_importance(_x_train, _y_train, model)

    return model


def train_tabnet(_x_train, _y_train, _x_test, _y_test, _sample_weight):
    def objective(trial):
        params = {
            "n_d": trial.suggest_int("n_d", 4, 16),
            "n_a": trial.suggest_int("n_a", 4, 16),
            "n_steps": trial.suggest_int("n_steps", 3, 10),
            "gamma": trial.suggest_float("gamma", 1.0, 2.0),
            "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-4, 1e-2, log=True),
            "optimizer_fn": torch.optim.Adam,
            "optimizer_params": {
                "lr": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True)  # Key change
            },        }

        model = TabNetClassifier(**params)
        model.fit(_x_train.values, _y_train.values, eval_set=[(_x_test.values, _y_test.values)], eval_metric=['auc'], weights=_sample_weight, max_epochs=50, patience=10)
        return max(model.history["val_0_auc"])  # Возвращаем accuracy на валидации

    # study = optuna.create_study(direction="maximize", pruner=MedianPruner())
    # study.optimize(objective, n_trials=50)
    #
    # print("Лучшие параметры:", study.best_params)

    # Задаём параметры модели
    tabnet_params = {
        "n_d": 14,              # Размерность шага предсказания
        "n_a": 11,              # Размерность шага внимания
        "n_steps": 10,          # Количество шагов
        "gamma": 1.87,          # Коэффициент масштабирования для шагов
        "lambda_sparse": 0.1, # Коэффициент разреженности
        "optimizer_fn": torch.optim.Adam,
        "optimizer_params": {"lr": 0.0002},
        "mask_type": "sparsemax",
        "device_name": "cuda" if torch.cuda.is_available() else "cpu",
    }

    # Создаём и обучаем модель
    clf = TabNetClassifier(**tabnet_params)
    clf.fit(
        X_train=_x_train.values,
        y_train=_y_train.values,
        eval_set=[(_x_test.values, _y_test.values)],
        eval_name=["valid"],
        max_epochs=50,
        patience=25,  # Ранняя остановка, если нет улучшений
        batch_size=1024,
    )

    # Предсказание
    y_pred = clf.predict(_x_test.values)
    # y_pred = (y_pred[:, 1] > 0.7).astype(int)
    conf_matrix = confusion_matrix(_y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    precision = precision_score(_y_test, y_pred)
    recall = recall_score(_y_test, y_pred)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

def train_random_forest_regr(_x_train, _y_train, _x_test, _y_test, _sample_weight, _num_iters):
    model = RandomForestRegressor(n_estimators=100, max_features='sqrt')
    best_precision = 0.
    best_model = model

    print("Model attr:", model.__dict__)
    test_result = {}
    print(f"Fitting Random Forest...")
    for iter in range(_num_iters):
        model.fit(_x_train, _y_train, sample_weight=_sample_weight)
        print("\nModel attr after fitting:", model.__dict__)
        predictions = model.predict(_x_test)

        # Transform probabilities to binary classification output in order to calc metrics:
        thrs = 0.5
        for i, p in enumerate(predictions):
            if p > thrs:
                predictions[i] = 1
            else:
                predictions[i] = 0
        print(_y_test, predictions)
        test_result['Precision'] = precision_score(_y_test, predictions)
        if test_result['Precision'] > best_precision:
            best_precision = test_result['Precision']
            best_model = model

            test_result['R2'] = r2_score(_y_test, predictions)
            test_result['Recall'] = recall_score(_y_test, predictions)
            test_result['F1'] = f1_score(_y_test, predictions)

    print(f"\nRandom Forest best result: R2 score = {test_result['R2']}\nRecall = {test_result['Recall']}\nPrecision = {test_result['Precision']}\nF1 = {test_result['F1']}")

    feature_importance = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': _x_train.columns, 'Importance': feature_importance})
    print(feature_importance_df)

    return best_model


def train_random_forest_cls(_x_train, _y_train, _x_test, _y_test, _sample_weight, _num_iters):
    model = RandomForestClassifier(n_estimators=200,
                                   max_depth=9,
                                   max_features=5,
                                   min_samples_leaf=5,
                                   min_samples_split=5)
    best_f1 = 0.
    best_model = model

    test_result = {}
    # grid_space = {'max_depth': [3, 5, 10, None],
    #               'n_estimators': [50, 100, 200],
    #               'max_features': [1, 3, 5, 7, 9],
    #               'min_samples_leaf': [1, 2, 3, 7],
    #               'min_samples_split': [1, 2, 3, 7]
    #               }
    # grid = GridSearchCV(model, param_grid=grid_space, cv=3, scoring='f1')
    # model_grid = grid.fit(_x_train, _y_train)
    # print('Best hyperparameters are: ' + str(model_grid.best_params_))
    # print('Best score is: ' + str(model_grid.best_score_))

    print(f"Fitting Random Forest classifier...")
    for iter in range(_num_iters):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(_x_train, _y_train)  #, sample_weight=_sample_weight)
        predictions = model.predict(_x_test)

        test_result['Precision'] = precision_score(_y_test, predictions)
        test_result['Recall'] = recall_score(_y_test, predictions)
        test_result['F1'] = f1_score(_y_test, predictions)

        if test_result['F1'] > best_f1:
            best_f1 = test_result['F1']
            best_model = model

            test_result['Recall'] = recall_score(_y_test, predictions)
            test_result['Precision'] = precision_score(_y_test, predictions)

    print(f"\nRandom Forest classifier best result: Recall = {test_result['Recall']}\nPrecision = {test_result['Precision']}\nF1 = {test_result['F1']}")

    feature_importance = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': _x_train.columns, 'Importance': feature_importance})
    print(feature_importance_df)

    return best_model



def train(_x_train, _y_train, _x_test, _y_test, _sample_weight, _cat_feats_encoded, _model_name, _num_iters, _maximize):
    if _model_name == 'XGBoostClassifier':
        model = train_xgboost_classifier(_x_train, _y_train, _x_test, _y_test, _sample_weight, _num_iters)
    elif _model_name == 'RandomForestRegressor':
        model = train_random_forest_regr(_x_train, _y_train, _x_test, _y_test, _sample_weight, _num_iters)
    elif _model_name == 'TabNet':
        model = train_tabnet(_x_train, _y_train, _x_test, _y_test, _sample_weight)
    elif _model_name == "RandomForestClassifier":
        model = train_random_forest_cls(_x_train, _y_train, _x_test, _y_test, _sample_weight, _num_iters)
    elif _model_name == "CatBoostClassifier":
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


def show_decision_tree(_model):
    tree_ = _model.estimators_[0].tree_
    feature_list = ["department_encoded",
                    "department_encoded2",
                    "department_encoded3",
                    "seniority_encoded",
                    "nationality_encoded",
                    "nationality_encoded2",
                    "nationality_encoded3",
                    "age_encoded",
                    "gender_encoded",
                    "gender_encoded2",
                    "gender_encoded3",
                    # "vacation_days_encoded",
                    "days_before_salary_increase_encoded",
                    "salary_increase_encoded",
                    "overtime_encoded",
                    "family_status_encoded",
                    "family_status_encoded2",
                    "family_status_encoded3",
                    "family_status_encoded4",
                    # "km_to_work_encoded",
                    "salary_6m_average_encoded",
                    "salary_cur_encoded"
                    ]
    feature_names = [feature_list[i] for i in tree_.feature]
    feature_name = [
        feature_names[i] for i in tree_.feature
    ]

    def recurse(node, depth):
        indent = "  " * depth
        if (tree_.feature[node] == -2):
            print("{}return {}".format(indent, tree_.value[node]))
        else:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)

    recurse(0, 1)


def test(_model, trn, trg):
    #_model = pickle.load(open("model_66_40_02_with_rains_but_seems_ok.pkl", 'rb'))
    y_proba = _model.predict_proba(trn)[:, 1]
    precision, recall, thresholds = precision_recall_curve(trg, y_proba)

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)

    # Находим порог с максимальным F1
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]

    thresholds = [optimal_threshold]  #"[optimal_threshold-0.1, optimal_threshold, optimal_threshold+0.1]  #[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    N_1 = len([y for y in trg if y == 1])
    N_0 = len([y for y in trg if y == 0])

    def adjusted_precision(P, N, M, new_N, new_M):
        numerator = P * (new_N / N)
        denominator = numerator + (1 - P) * (new_M / M)
        return numerator / denominator

    for thrs in thresholds:
        print(f"Testing on united data with threshold = {thrs}...")
        predictions = _model.predict_proba(trn)
        predictions = (predictions[:, 1] > thrs).astype(int)

        f1_united = f1_score(trg, predictions)
        recall_united = recall_score(trg, predictions)
        precision_united = precision_score(trg, predictions)
        precision_united = adjusted_precision(precision_united, N_1, N_0, 1600, 8400)
        print(f"CatBoost result: F1 = {f1_united:.2f}, Recall = {recall_united:.2f}, Precision - {precision_united:.2f}")
        result = confusion_matrix(trg, predictions, normalize='true')
        sn.set(font_scale=1.4)  # for label size
        sn.heatmap(result, annot=True, annot_kws={"size": 16})  # font size

        plt.show()
        plt.clf()
    return f1_united, recall_united, precision_united

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


def main(_config: dict):
    data_path = _config['dataset_src']
    datasets = collect_datasets(data_path)
    rand_states = [4] # range(5)  # [777, 42, 6, 1370, 5087]
    score = [0, 0, 0]

    for c in datasets[0].columns:
            if 'Credit' in c:
                if 'Надежность' in c or 'before' not in c:
                    _config['cat_features'] += [c]

    if _config['calculated_features']:
        datasets, new_cat_feat = create_features_for_datasets(datasets)
        _config['cat_features'] += new_cat_feat

    for split_rand_state in rand_states:
        d_train, d_test, cat_feats_encoded = prepare_dataset_2(datasets, _config['normalize'], _config['make_synthetic'], _config['encode_categorical'], _config['cat_features'], split_rand_state)
        d_test = d_test.drop(columns=['INN', 'level_0', 'index'], errors='ignore')
        d_train = d_train.drop(columns=['INN', 'level_0', 'index'], errors='ignore')

        print(f"X train: {d_train.shape[0]}, x_val: {d_test.shape[0]}")

        if _config['smote']:
            d_train = minority_class_resample(d_train,cat_feats_encoded)
        #
        # d_train = d_train.drop(columns=['total_spacetime_area'])
        # d_test = d_test.drop(columns=['total_spacetime_area'])


        x_train = d_train.drop('ACTIVITY_AND_ATTRITION', axis=1)
        y_train = d_train['ACTIVITY_AND_ATTRITION']
        x_val = d_test.drop('ACTIVITY_AND_ATTRITION', axis=1)
        y_val = d_test['ACTIVITY_AND_ATTRITION']

        # # Считаем корреляции с таргетом
        # correlations = x_train.corrwith(d_train['ACTIVITY_AND_ATTRITION']).abs().sort_values(ascending=False)
        #
        # # Визуализация
        # sn.barplot(x=correlations.head(20), y=correlations.head(20).index)
        # plt.title('Top 20 features correlated with target')
        # plt.show()


        print(f"X train: {x_train.shape[0]}, x_val: {x_val.shape[0]}, y_train: {y_train.shape[0]}, y_val: {y_val.shape[0]}")
        sample_weight = calc_weights(y_train, y_val)

        # print(sample_weight)
        trained_model = train(x_train, y_train, x_val, y_val, sample_weight, cat_feats_encoded, _config['model'], _config['num_iters'], _config['maximize'])
        print('Metrics on TRAIN set:')
        _, _, _ = test(trained_model, x_train, y_train)
        print('Metrics on TEST set:')
        f1, r, p = test(trained_model, x_val, y_val)
        score[0] += f1
        score[1] += r
        score[2] += p

        if _config['model'] == 'RandomForestClassifier':
           show_decision_tree(trained_model)

    score = (score[0] / len(rand_states), score[1] / len(rand_states), score[2] / len(rand_states))
    print(f"Final score of cross-val: F1={score[0]:.2f}, Recall = {score[1]:.2f}, Precision={score[2]:.2f}")

    with open('model.pkl', 'wb') as f:
       print("Saving model..")
       pickle.dump(trained_model, f)


if __name__ == '__main__':
    # config_path = 'config.json'  # config file is used to store some parameters of the dataset
    config = {
        'model': 'CatBoostClassifier',  # options: 'TabNet', 'RandomForestClassifier', 'XGBoostClassifier', 'CatBoostClassifier'
        'num_iters': 10,
        'normalize': False,  # normalize input values or not
        'maximize': 'Precision',  # metric to maximize
        'dataset_src': 'data/v12',
        'encode_categorical': True,
        'calculated_features': True,
        'make_synthetic': None,  # options: 'sdv', 'ydata', None
        'smote': False,  # perhaps not needed for catboost and in case if minority : majority > 0.5
        'cat_features': ['Seasonality', 'legal_type']  # , 'legal_type']  # , 'DRIVER_FIO', 'Entity Type', 'taxcode']  #, 'kbktax', 'kbknametax']  #, 'occupational_hazards']
    }

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


Would you like me to refine any of these based on a specific business goal (e.g., predicting churn, maximizing revenue, or improving engagement)?"""