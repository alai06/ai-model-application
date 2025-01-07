import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model_path, X_test_path, y_test_path):
    model = load_model(model_path)
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)

    # Faire des prédictions
    y_pred = model.predict(X_test)

    # Calculer les métriques
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")

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

    # Tracer les métriques
    plt.figure(figsize=(8, 6))
    plt.bar(['MSE', 'R²'], [mse, r2], color=['orange', 'blue'])
    plt.title("Métriques d\'évaluation")
    plt.ylabel("Valeur")
    plt.savefig("img/metrics_plot.png")
    plt.close()

if __name__ == "__main__":
    evaluate_model(
        model_path="data/trained_model.h5",
        X_test_path="data/X_test.npy",
        y_test_path="data/y_test.npy"
    )
