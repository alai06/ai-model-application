import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.model import build_lstm_model

def create_time_windows(data, window_size, cols_to_use):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data.iloc[i:i+window_size][cols_to_use].values)
        y.append(data.iloc[i+window_size]['Close'])
    return np.array(X), np.array(y)

def train_model(data_path, model_save_path, X_test_save_path, y_test_save_path):
    data = pd.read_csv(data_path, index_col="Date")

    # Création des fenêtres temporelles
    window_size = 3
    cols_to_use = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    X, y = create_time_windows(data, window_size, cols_to_use)

    # Division des ensembles
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Construire et entraîner le modèle
    model = build_lstm_model(input_shape=(window_size, len(cols_to_use)))
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

    # Sauvegarder le modèle et les ensembles
    model.save(model_save_path)
    np.save(X_test_save_path, X_test)
    np.save(y_test_save_path, y_test)

if __name__ == "__main__":
    train_model(
        data_path="data/preprocessed_data.csv",
        model_save_path="data/trained_model.h5",
        X_test_save_path="data/X_test.npy",
        y_test_save_path="data/y_test.npy"
    )
