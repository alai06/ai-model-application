from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import regularizers

def build_lstm_model(input_shape, lstm_units=50, dropout_rate=0.2, regularization=0.01):
    model = Sequential([
        LSTM(lstm_units, input_shape=input_shape, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(lstm_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(1, activation='linear', kernel_regularizer=regularizers.L2(regularization))  # Sortie unique avec L2
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
