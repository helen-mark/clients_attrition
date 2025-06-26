import tensorflow as tf
import tensorflow.keras as K
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

import pandas as pd
import os
import numpy as np
from imblearn.over_sampling import SMOTENC

# from ydata.connectors import LocalConnector
# from ydata.metadata import Metadata
# from ydata.synthesizers.regular.model import RegularSynthesizer
# from ydata.dataset.dataset import Dataset
# from ydata.report import SyntheticDataProfile

from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, CopulaGANSynthesizer


def dataframe_to_dataset(dataframe: pd.DataFrame):
    dataframe = dataframe.copy()
    labels = dataframe.pop("status")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


def encode_numerical_feature(feature: K.Input, name: str, dataset: tf.data.Dataset):  # numerical features like age, salary etc
    # Create a Normalization layer for our feature
    normalizer = K.layers.Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature


def encode_categorical_feature(feature: K.Input, name: str, dataset: tf.data.Dataset, is_string: bool):  # categorical features like sex, family status etc
    lookup_class = K.layers.StringLookup if is_string else K.layers.IntegerLookup
    lookup = lookup_class(output_mode="binary")

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    lookup.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = lookup(feature)
    return encoded_feature

def minority_class_resample(_dataset: pd.DataFrame, _cat_feat: list):
    print("Smote application", _dataset.columns)
    categorical_indices = [i for i, col in enumerate(_dataset.columns) if col in _cat_feat]
    smote = SMOTENC(categorical_features=categorical_indices, sampling_strategy='auto', k_neighbors=5, random_state=42)
    new_datasets = []
    X = _dataset.drop('ACTIVITY_AND_ATTRITION', axis=1)
    y = _dataset['ACTIVITY_AND_ATTRITION']

    for n, row in X.iterrows():
        if row.isnull().values.any():
            print(f"NaN value in snapshot dataset: {row}")
    X_resampled, y_resampled = smote.fit_resample(X, y)
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled['ACTIVITY_AND_ATTRITION'] = y_resampled
    return df_resampled

def collect_datasets(_data_path: str):
    datasets = []
    for filename in sorted(os.listdir(_data_path)):
        if '.csv' not in filename:
            continue
        dataset_path = os.path.join(_data_path, filename)
        print(filename)
        dataset = pd.read_csv(dataset_path)
        print("cols before", dataset.columns)
        strings_to_drop = ['City', 'latest', 'Latest', 'First_date', 'First date', 'start_date', 'ДАТА', 'upper_b',
                          #'SQM_SINGLE_MATS_IN_ACTIVE_SPECIFICATIONS',
                           'sum_recalculations', '_sum_0', '_avg_0',# 'address',
                        'aaaaaaaaaasqm_sum']  #, 'sqm_mean']
        dataset = dataset.drop(
            columns=[c for c in dataset.columns if any(string in c for string in strings_to_drop)])
        print("cols after", dataset.columns)
        dataset = shuffle(dataset, random_state=4242)
        datasets.append(dataset)
    return datasets


def DataFilter(x, y):  # remove samples with 0 days before salary increase (as an experiment)
    return x['days_before_salary_increase'] > 0


def create_dataset(_dataset: pd.DataFrame, _dataset_val: pd.DataFrame, _batch_size: int):
    print(
        f"Using {len(_dataset)} samples for training "
        f"and {len(_dataset_val)} for validation"
    )

    train_ds = dataframe_to_dataset(_dataset)
    val_ds = dataframe_to_dataset(_dataset_val)
    val_ds_2 = dataframe_to_dataset(_dataset_val)
    print("B", len(list(train_ds.as_numpy_iterator())))

    # train_ds = train_ds.filter(DataFilter)
    # val_ds = val_ds.filter(DataFilter)
    # val_ds_2 = val_ds.filter(DataFilter)


    print("A", len(list(train_ds.as_numpy_iterator())))

    train_ds = train_ds.batch(_batch_size)
    val_ds = val_ds.batch(_batch_size)
    return train_ds, val_ds, val_ds_2, _dataset, _dataset_val

def make_synthetic(_dataset: pd.DataFrame, _size: int, _type: str = 'sdv'):
    if _type == 'sdv':
        return make_sdv(_dataset, _size)
    else:
        return make_ydata(_dataset, _size)

def make_ydata(_dataset: pd.DataFrame, _size: int = 1000):
    # connector = LocalConnector()  # , keyfile_dict=token)
    #
    # # Instantiate a synthesizer
    # cardio_synth = RegularSynthesizer()
    # # memory_usage = trn.memory_usage(index=True, deep=False).sum()
    # # npartitions = int(((memory_usage / 10e5) // 100) + 1)
    # data = Dataset(_dataset)
    # # calculating the metadata
    # metadata = Metadata(data)
    #
    # # fit model to the provided data
    # cardio_synth.fit(data, metadata, condition_on=["status"])
    #
    # # Generate data samples by the end of the synth process
    # synth_sample = cardio_synth.sample(n_samples=_size,
    #                                    # condition_on={
    #                                    #  "status": {
    #                                    #      "categories": [{
    #                                    #          "category": 0,
    #                                    #          "percentage": 1.0
    #                                    #      }]
    #                                    #  }}
    #                                    )
    #
    # # TODO target variable validation
    # profile = SyntheticDataProfile(
    #     data,
    #     synth_sample,
    #     metadata=metadata,
    #     target="status",
    #     data_types=cardio_synth.data_types)
    #
    # profile.generate_report(
    #     output_path="./cardio_report_example.pdf",
    # )
    # return synth_sample.to_pandas()  # {t: df.to_pandas() for t, df in res.items()}

    pass

def make_sdv(_dataset: pd.DataFrame, _size: int = 1000):
    metadata = Metadata.detect_from_dataframe(
        data=_dataset,
        table_name='attrition')
    synthesizer = GaussianCopulaSynthesizer(metadata=metadata,
                                            # numerical_distributions={
                                            #     'income_shortterm': 'gamma',
                                            #     'age': 'truncnorm',
                                            #     'absenteeism_shortterm': 'gamma',
                                            #     'seniority': 'norm',
                                            #     'vacation_days_shortterm': 'gamma'
                                            # }
                                            )

    df = _dataset.copy()
    synthesizer = CTGANSynthesizer(metadata=metadata,
                                   epochs=100,  # Increase from default 300 if needed
                                   batch_size=100,  # (len(df) // 4) // 10 * 10 ,  # ~25% of dataset size
                                   generator_dim=(128, 128),  # Larger networks for complex relationships
                                   discriminator_dim=(128, 128),
                                   verbose=True,  # To monitor training progress
                                   pac=5,  # Helps with mode collapse for categoricals
                                   cuda=True
                                   )

    #synthesizer = TVAESynthesizer(metadata=metadata)
    for col in df.select_dtypes(include=['category']).columns:
        df[col] = df[col].astype('object')
    synthesizer.fit(df)

    synthetic_data = synthesizer.sample(num_rows=_size)
    return synthetic_data


def encode_categorical(_dataset: pd.DataFrame, _encoder: OneHotEncoder, _cat_features: list):
    encoded_features = _encoder.transform(_dataset[_cat_features]).toarray().astype(int)
    encoded_df = pd.DataFrame(encoded_features, columns=_encoder.get_feature_names_out(_cat_features))
    numerical_part = _dataset.drop(columns=_cat_features)
    return pd.concat([encoded_df, numerical_part], axis=1), encoded_df


def get_united_dataset(_d_train: list, _d_val: list, _d_test: list):
    trn = pd.concat(_d_train, axis=0)
    vl = pd.concat(_d_val, axis=0)

    trn = trn.transpose()
    vl = vl.transpose()

    x_train = trn[:-1].transpose()
    x_val = vl[:-1].transpose()
    y_train = trn[-1:].transpose()
    y_val = vl[-1:].transpose()
    return x_train, y_train, x_val, y_val

def get_split(_dataset: pd.DataFrame, _test_split: float, _split_state: int):
    # val = _dataset.sample(frac=_test_split, random_state=_split_state)
    # test = val  # .sample(frac=0.3, random_state=_split_rand_state)
    # # val = val.drop(test.index)
    # trn = _dataset.drop(val.index)

    n_splits = 5
    fold_size = len(_dataset) // n_splits

    start_index = _split_state * fold_size
    end_index = (_split_state + 1) * fold_size if _split_state < n_splits - 1 else len(_dataset)

    trn = pd.concat([_dataset.iloc[:start_index], _dataset.iloc[end_index:]])
    val = _dataset.iloc[start_index:end_index]
    test = val

    return trn, val, test


def normalize(_data: pd.DataFrame, cat_features: list = None):
    if cat_features is None:
        cat_features = []

    # Identify numerical columns (exclude categorical features)
    num_features = [col for col in _data.columns if col not in cat_features + ['ACTIVITY_AND_ATTRITION']]

    # Normalize only numerical columns
    _data_normalized = _data.copy()
    _data_normalized[num_features] = _data[num_features].apply(
        lambda x: (x - x.min()) / (x.max() - x.min()),
        axis=0
    )
    return _data_normalized

def prepare_dataset_2(_datasets: list, _normalize: bool, _make_synthetic: bool, _encode_categorical: bool, _cat_feat: list, _split_rand_state: int):
    if _encode_categorical:
        concat_dataset = pd.concat(_datasets)
        encoder = OneHotEncoder()
        encoder.fit(concat_dataset[_cat_feat])

    # if _make_synthetic is not None:
    #     united = pd.concat(_datasets, axis=0)
    #     sample_df = make_synthetic(united,1000)
    #
    #     # Check for similar rows and print if match:
    #     for n, syn_row in united.iterrows():
    #         for n1, real_row in sample_df.iterrows():
    #             if (syn_row.values.astype(int) == real_row.values.astype(int)).all():
    #                 print("Match!", real_row.values.astype(int), syn_row.values.astype(int))
    #
    #     if _encode_categorical:
    #         sample_df, _ = encode_categorical(sample_df, encoder)

    trn1 = _datasets[1]
    trn2 = _datasets[2]
    #trn3 = _datasets[3]
    tst = _datasets[0]

    if _normalize:
        print('Performing normalization...')
        trn1 = normalize(trn1, _cat_feat)
        trn2 = normalize(trn2, _cat_feat)
        #trn3 = normalize(trn3, _cat_feat)

        tst = normalize(tst, _cat_feat)

    trn = pd.concat([trn1, trn2]).reset_index().drop(columns=['index'])
    rows_1 = trn[trn['ACTIVITY_AND_ATTRITION'] == 1]
    rows_2 = trn[trn['ACTIVITY_AND_ATTRITION'] == 0].sample(n=2000)
    trn = pd.concat([rows_1, rows_2]).reset_index()
    # print('TO REMOVE\n', rows_to_remove)
    # trn = trn.drop(rows_to_remove.index)
    trn = shuffle(trn).drop(columns=['index'])


    print('Train dataset concat, normalized:\n', trn)

    if _make_synthetic is not None:
      sample_trn = make_synthetic(trn, len(trn)//3)
      if _encode_categorical:
          sample_trn, _ = encode_categorical(sample_trn, encoder, _cat_feat)

    if _encode_categorical:
        # Convert the encoded features to a DataFrame
        tst, _ = encode_categorical(tst, encoder, _cat_feat)
        trn, encoded_part = encode_categorical(trn, encoder, _cat_feat)
        # if _make_synthetic is not None:
        #     sample_trn, _ = encode_categorical(sample_trn, encoder, _cat_feat)
        cat_feature_names = encoded_part.columns.values.tolist()

        # if _make_synthetic is not None:
        #     trn = pd.concat([trn, sample_trn], axis=0)

    if _make_synthetic:
        trn = pd.concat([trn, sample_trn], axis=0)
    return trn, tst, cat_feature_names


def add_quality_features(df: pd.DataFrame, _total_ds: pd.DataFrame):
    print(df.columns)
    # df['total_spacetime_area_normalized'] = df['total_spacetime_area'] * 100 / df['total_spacetime_area'].sum()
    # df = df.drop(columns=['total_spacetime_area'])
    df['total_spacetime_area'] = df['total_spacetime_area'] / df['Active_months']
    df['total_spacetime_area_normalized'] = df['total_spacetime_area'] * 100 / df['total_spacetime_area'].sum()

    df['Turnover_sum_last_12'] = df['Turnover_sum_last_12'] / df['Active_months']
    df['sqm_sum'] = df['sqm_sum'] / df['Active_months']
    #df['sum_recalculations'] = df['sum_recalculations'] / df['Active_months']
    df['Frequency_of_changes_avg'] = df['Frequency_of_changes_sum'] / df['Active_months']
    df['total_recalculations'] = df['total_recalculations'] / df['Active_months']
    # df['drivers_per_address'] = df['n_drivers_per_12'] / df['unique_addresses_last_12']

    #df['turnover_group'] = pd.qcut(_total_ds['Turnover_sum_last_12'], q=5, labels=['low', 'medium_low', 'medium', 'medium_high', 'high'], duplicates='drop')
    #df['turnover_vs_seasonality'] = df['turnover_group'].astype('str') + '_' + df['Seasonality'].astype('str')
    #df['carpet_area'] = df['sqm_sum'] / (df['Frequency_of_changes'])
    df['tenure_adjusted_activity'] = df['Active_months'] / df['Dur_months']
    calculated_cat_feat = []  #'turnover_group', 'turnover_vs_seasonality']

    return df, calculated_cat_feat

def create_features_for_datasets(_datasets: list):
    improved_datasets = []
    tot_ds = pd.concat(_datasets, axis=0)
    for d in _datasets:
        print('create features for next dataset...')
        d, new_cat_feat = add_quality_features(d, tot_ds.reset_index())
        cols = d.columns.tolist()
        d = d[cols]
        improved_datasets.append(d)
    return improved_datasets, new_cat_feat