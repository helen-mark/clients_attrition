"""
  Created on Jan 2025
@author: Elena Markova
          for Attrition Rate Project
"""

import pickle
from pandas import read_csv
from tensorflow import keras as K
from sklearn.metrics import r2_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder

import pandas as pd
import os
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from utils.dataset_13 import collect_datasets, normalize, create_features_for_datasets, encode_categorical


def test_model(_model: K.Model, _feat: pd.DataFrame, _trg: pd.DataFrame):
    predictions = model.predict_proba(_feat)
    predictions = [int(p[1] > 0.52) for p in predictions]

    f1 = f1_score(_trg, predictions)
    r = recall_score(_trg, predictions)
    p = precision_score(_trg, predictions)

    print(f"F1 = {f1:.2f}, Recall = {r:.2f}, Precision = {p:.2f}")
    result = confusion_matrix(_trg, predictions)

    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(result, annot=True, annot_kws={"size": 16}, fmt='d')  # font size

    plt.show()


def test_rowwise(_model, _dataset: pd.DataFrame):
    # Filter to only active clients
    active_mask = _dataset['ACTIVITY_AND_ATTRITION'] == 0
    active_clients = _dataset #[active_mask]
    gone_clients = _dataset[~active_mask]

    if len(gone_clients) > 0:
        print(f"{len(gone_clients)} clients are gone")

    # Prepare features for active clients
    features = active_clients.drop(columns=['INN', 'ACTIVITY_AND_ATTRITION'])

    print(len(features.columns))

    # Predict all active clients at once
    probabilities = _model.predict_proba(features)
    predictions = (probabilities[:, 1] > 0.215).astype(int)

    # Build result dictionary (only for active clients)
    result = {
        'INN': active_clients['INN'].tolist(),
        'Prediction': predictions.tolist(),
        'Status': active_clients['ACTIVITY_AND_ATTRITION'].tolist()
    }

    return result


if __name__ == '__main__':
    config = {
        "control_data_path": "data/control_v20/",
        "data_path": "data/v21/",
        "test_data_path": "data/test_v20",
        "model_path": "model.pkl",
        "normalize": False,
        "cat_features": ['Seasonality', 'legal_type']
    }

    model = pickle.load(open(config["model_path"], 'rb'))
    datasets = collect_datasets(config["data_path"])
    test_datasets = collect_datasets(config['test_data_path'])

    datasets_with_fetures, cat_feats = create_features_for_datasets(test_datasets+datasets, config)
    test_dataset = datasets_with_fetures[0]
    train_datasets = datasets_with_fetures[1:]
    config['cat_features'] += cat_feats
    print(cat_feats)

    concat_dataset = pd.concat(datasets_with_fetures, axis=0)
    encoder = OneHotEncoder()
    encoder.fit(concat_dataset[config['cat_features']])
    concat_dataset = concat_dataset.reset_index()
    datasets.append(concat_dataset)
    concat_dataset, cat_feature_names = encode_categorical(concat_dataset, encoder, config['cat_features'])

    if config['normalize']:
        test_dataset = normalize(test_dataset, config['cat_features'])

    test_dataset, cat_feature_names = encode_categorical(test_dataset, encoder, config['cat_features'])

#    test_model(model, test_dataset.drop(columns=['ACTIVITY_AND_ATTRITION', 'INN']), test_dataset['ACTIVITY_AND_ATTRITION'])

    # feat.insert(12, 'occupational_hazards_4', [0 for i in range(len(feat))])
    result = test_rowwise(model, test_dataset)
    result_df = pd.DataFrame(result)
    print(result_df)
    result_df.to_excel('predictions_merch.xlsx', index=False)