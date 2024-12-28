import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    # Normalisation
    scaler = MinMaxScaler(feature_range=(-1, 1))
    cols_to_normalize = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])

    # Regroupement mensuel
    monthly_data = df.groupby(['Year', 'Month'])[cols_to_normalize].mean().reset_index()
    monthly_data['Date'] = pd.to_datetime(
        monthly_data['Year'].astype(str) + '-' + monthly_data['Month'].astype(str) + '-01'
    )
    monthly_data.set_index('Date', inplace=True)
    return monthly_data

if __name__ == "__main__":
    file_path = "data/BTC-USD.csv"
    data = load_and_preprocess_data(file_path)
    data.to_csv("data/preprocessed_data.csv")
