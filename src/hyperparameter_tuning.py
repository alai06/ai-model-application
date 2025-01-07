import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_model(optimizer='adam', lstm_units=50, dropout_rate=0.2):
    model = Sequential([
        LSTM(lstm_units, input_shape=(window_size, len(cols_to_use)), return_sequences=True),
        Dropout(dropout_rate),
        Dense(1)
    ])
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Charger les données prétraitées
data = pd.read_csv("data/preprocessed_data.csv", index_col="Date")

# Préparer les données pour l'entraînement
window_size = 3
cols_to_use = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

def create_time_windows(data, window_size, cols_to_use):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data.iloc[i:i+window_size][cols_to_use].values)
        y.append(data.iloc[i+window_size]['Close'])
    return np.array(X), np.array(y)

X, y = create_time_windows(data, window_size, cols_to_use)

# Définir le modèle pour GridSearchCV
model = KerasClassifier(build_fn=build_model, verbose=0)

# Définir les hyperparamètres à tester
params = {
    'optimizer': ['adam', 'rmsprop'],
    'lstm_units': [50, 100],
    'dropout_rate': [0.1, 0.2, 0.3],
    'batch_size': [16, 32],
    'epochs': [10, 20]
}

# Effectuer la recherche d'hyperparamètres
grid = GridSearchCV(estimator=model, param_grid=params, cv=3, n_jobs=-1)
grid_result = grid.fit(X, y)

# Afficher les meilleurs hyperparamètres
print(f"Best parameters: {grid_result.best_params_}")
