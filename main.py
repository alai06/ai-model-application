from pre.data_preprocessing import load_and_preprocess_data
from src.train_model import train_model
from src.evaluate_model import evaluate_model

def main():
    # Prétraitement des données
    print("Prétraitement des données...")
    data = load_and_preprocess_data("data/BTC-USD.csv")
    data.to_csv("data/preprocessed_data.csv")
    print("Données prétraitées et enregistrées.")

    # Entraînement du modèle
    print("Entraînement du modèle...")
    train_model(
        data_path="data/preprocessed_data.csv",
        model_save_path="data/trained_model.h5",
        X_test_save_path="data/X_test.npy",
        y_test_save_path="data/y_test.npy"
    )
    print("Modèle entraîné et sauvegardé.")

    # Évaluation du modèle
    print("Évaluation du modèle...")
    evaluate_model(
        model_path="data/trained_model.h5",
        X_test_path="data/X_test.npy",
        y_test_path="data/y_test.npy"
    )
    print("Évaluation terminée.")

if __name__ == "__main__":
    main()
