import datetime
import os

from tqdm import tqdm  # for progress bar

import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import save_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

data_path = 'data/v8/'

def collect_data():
    all_df = []
    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)
        df = pd.read_csv(file_path)
        all_df.append(df)
    raw_data = pd.concat(all_df)
    raw_data = raw_data.sort_values(by='INN')
    return raw_data


def prepare_lstm_data(df, inn_column='INN', target_column='ACTIVITY_AND_ATTRITION', feature_columns=None):
    """
    Convert DataFrame with client monthly data to LSTM-ready format

    Args:
        df: DataFrame containing the data
        inn_column: Name of the column with client identifiers (INN)
        target_column: Name of the target column
        feature_columns: List of feature columns to include (None = all except INN and target)

    Returns:
        clients_data: List of numpy arrays with client sequences (features)
        clients_targets: List of numpy arrays with corresponding targets
        feature_names: List of feature names used
    """
    drop_columns = ['city', 'City', 'First_date', 'first_date', 'start_date', 'upper_bound', 'latest_date', 'Latest_date', 'Seasonality', 'legal_type', 'ДАТА КОНТРАКТА']
    # Prepare feature columns if not specified
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col not in [inn_column, target_column] + drop_columns]
    print(f'Feature columns: {feature_columns}')

    # Sort by INN and date to ensure chronological order
    df = df.sort_values([inn_column, 'upper_bound'])

    # Get unique clients
    unique_inns = df[inn_column].unique()

    clients_data = []
    clients_targets = []

    print("Processing client sequences...")
    for inn in tqdm(unique_inns):
        # Get all rows for this client
        client_df = df[df[inn_column] == inn]

        # Extract features and target
        features = client_df[feature_columns].values.astype('float32')
        targets = client_df[target_column].values.astype('float32')

        # Only include clients with at least 5 time steps (months)
        if len(features) >= 5:
            clients_data.append(features)
            clients_targets.append(targets)

    return clients_data, clients_targets, feature_columns


def main():
    raw_data = collect_data()

    # Prepare the data
    clients_data, clients_targets, feature_names = prepare_lstm_data(raw_data)

    # Print summary
    print(f"\nPrepared data for {len(clients_data)} clients")
    print(
        f"Sequence lengths range from {min(len(x) for x in clients_data)} to {max(len(x) for x in clients_data)} months")
    print(f"Using features: {feature_names}")

    # Sample output for first client
    print("\nSample client data (first 3 months):")
    print(clients_data[0][:3])
    print("Corresponding targets:")
    print(clients_targets[0][:3])

    # Pad sequences to the same length (with zeros or other mask value)
    max_len = max(len(seq) for seq in clients_data)
    X = pad_sequences(clients_data, maxlen=max_len, dtype='float32', padding='post', truncating='post', value=-2000000.)
    y = pad_sequences(clients_targets, maxlen=max_len, dtype='float32', padding='post', truncating='post', value=-2000000.)
    y = y.reshape(y.shape[0], y.shape[1], 1)
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    # Batch the data
    batch_size = 128
    train_dataset = train_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    model = Sequential([
        # Masking layer to ignore padded timesteps
        Masking(mask_value=-2000000., input_shape=(max_len, X.shape[2])),
        LSTM(64, return_sequences=True),  # Return sequences for each timestep
        Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['recall', 'precision'])

    log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir,
                                       histogram_freq=1,
                                       profile_batch='500,520')

    model.fit(train_dataset, epochs=10,
              validation_data=test_dataset,
              callbacks=[tensorboard_callback]
              )

    model_save_path = 'client_attrition_lstm_model.h5'
    save_model(model, model_save_path)
    print(f"Model saved to {model_save_path}")
    # model = load_model(model_save_path)

    # Evaluate on test set
    test_loss, test_rec, test_prec = model.evaluate(X_test, y_test, batch_size=32)
    print(f"\nTest Recall: {test_rec:.4f}, Test Precision: {test_prec:.4f}")

    # Generate predictions
    y_pred = model.predict(X_test)

    # Convert probabilities to class labels
    y_pred_classes = (y_pred > 0.5).astype("int32")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test.flatten(), y_pred_classes.flatten()))

if __name__ == '__main__':
    main()