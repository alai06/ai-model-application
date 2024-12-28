import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def evaluate_model(model_path, X_test_path, y_test_path):
    model = load_model(model_path)
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)

    # Faire des prédictions
    y_pred = model.predict(X_test)

    # Visualiser les résultats
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Réel', color='blue')
    plt.plot(y_pred, label='Prédit', color='orange')
    plt.title("Comparaison des valeurs réelles et prédites")
    plt.xlabel("Index")
    plt.ylabel("Prix normalisé")
    plt.legend()
    plt.grid()
    plt.savefig("img/predictions_plot.png")  # Sauvegarde
    plt.close()  # Fermer la figure pour éviter les problèmes

if __name__ == "__main__":
    evaluate_model(
        model_path="data/trained_model.h5",
        X_test_path="data/X_test.npy",
        y_test_path="data/y_test.npy"
    )
