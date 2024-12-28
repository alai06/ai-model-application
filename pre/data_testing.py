import pandas as pd

def test_data(file_path):
    df = pd.read_csv(file_path)

    # Vérification des colonnes manquantes
    print("Colonnes avec données manquantes :")
    print(df.isnull().sum())

    # Aperçu des données
    print("Aperçu des premières lignes :")
    print(df.head())

if __name__ == "__main__":
    file_path = "data/BTC-USD.csv"
    test_data(file_path)
