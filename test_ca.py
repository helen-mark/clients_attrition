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
from sklearn.model_selection import train_test_split, cross_val_score
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
from catboost import CatBoostClassifier, Pool

from pathlib import Path
# from examples.local import setting_dask_env

import seaborn as sn
import matplotlib.pyplot as plt

from utils.dataset import create_features_for_datasets, collect_datasets, minority_class_resample, prepare_dataset_2, get_united_dataset

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

def test(_model, trn, trg):
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    for thrs in thresholds:
        print(f"Testing on united data with threshold = {thrs}...")
        predictions = _model.predict_proba(trn)
        predictions = (predictions[:, 1] > thrs).astype(int)

        f1_united = f1_score(trg, predictions)
        recall_united = recall_score(trg, predictions)
        precision_united = precision_score(trg, predictions)
        print(f"CatBoost result: F1 = {f1_united:.2f}, Recall = {recall_united:.2f}, Precision - {precision_united:.2f}")
        result = confusion_matrix(trg, predictions, normalize='true')
        sn.set(font_scale=1.4)  # for label size
        sn.heatmap(result, annot=True, annot_kws={"size": 16})  # font size

        plt.show()
        plt.clf()
    return

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

    if _config['calculated_features']:
        datasets, new_cat_feat = create_features_for_datasets(datasets)
        _config['cat_features'] += new_cat_feat


    for split_rand_state in rand_states:
        d_train, d_test, cat_feats_encoded = prepare_dataset_2(datasets, _config['normalize'], _config['make_synthetic'], _config['encode_categorical'], _config['cat_features'], split_rand_state)
        print(d_test)
        d_test = d_test.drop(columns='INN')
        d_train = d_train.drop(columns='INN')

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

        # correlations = x_train.corrwith(d_train['ACTIVITY_AND_ATTRITION']).abs().sort_values(ascending=False)
        #
        # sn.barplot(x=correlations.head(20), y=correlations.head(20).index)
        # plt.title('Top 20 features correlated with target')
        # plt.show()

        print(f"X train: {x_train.shape[0]}, x_val: {x_val.shape[0]}, y_train: {y_train.shape[0]}, y_val: {y_val.shape[0]}")
        trained_model = pickle.load(open("models/model_Rec_66_Prec_49_threshold_02.pkl", 'rb'))

        print('Metrics on TRAIN set:')
        test(trained_model, x_train, y_train)
        print('Metrics on TEST set:')
        test(trained_model, x_val, y_val)


if __name__ == '__main__':
    # config_path = 'config.json'  # config file is used to store some parameters of the dataset
    config = {
        'model': 'CatBoostClassifier',  # options: 'TabNet', 'RandomForestClassifier', 'XGBoostClassifier', 'CatBoostClassifier'
        'num_iters': 10,
        'normalize': False,  # normalize input values or not
        'maximize': 'Precision',  # metric to maximize
        'dataset_src': 'data/v7',
        'encode_categorical': True,
        'calculated_features': True,
        'make_synthetic': None,  # options: 'sdv', 'ydata', None
        'smote': False,  # perhaps not needed for catboost and in case if minority : majority > 0.5
        'cat_features': ['Seasonality', 'legal_type']  #, 'occupational_hazards']
    }

    main(config)