import torch, json
from os import system
import matplotlib.pyplot as plt
import numpy as np
from modele import SimpleModel
from create_data import prepare_json_with_lookback

def calculate_error_percentage(real, predicted):
    """
    Calcule l'erreur relative entre la valeur réelle et la valeur prédite.
    """
    return abs((real - predicted) / real) * 100
def evaluate_model(model_path, json_file, lookback):
    """
    Charge le modèle et les données JSON, effectue les prédictions, 
    calcule les statistiques, et génère les graphiques.
    """
    # Charger le modèle
    model = SimpleModel(lookback)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Charger les données JSON
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Variables pour collecter les stats
    real_highs = []
    predicted_highs = []
    real_lows = []
    predicted_lows = []
    real_third = []  # Pour la troisième sortie
    predicted_third = []
    real_fourth = []  # Pour la quatrième sortie
    predicted_fourth = []
    
    high_errors = []
    low_errors = []
    third_errors = []
    fourth_errors = []

    # Boucle sur toutes les données
    for entry in data:
        input_data = torch.tensor([entry['input']], dtype=torch.float32)
        real_high, real_low, real_third_value, real_fourth_value = entry['output']  # Les 4 sorties
        prediction = model(input_data).detach().numpy()[0]
        predicted_low, predicted_high, predicted_third_value, predicted_fourth_value = prediction

        # Collecter les données pour les stats
        real_highs.append(real_high)
        predicted_highs.append(predicted_high)
        real_lows.append(real_low)
        predicted_lows.append(predicted_low)
        real_third.append(real_third_value)
        predicted_third.append(predicted_third_value)
        real_fourth.append(real_fourth_value)
        predicted_fourth.append(predicted_fourth_value)

        # Calcul des erreurs
        high_error = calculate_error_percentage(real_high, predicted_high)
        low_error = calculate_error_percentage(real_low, predicted_low)
        third_error = calculate_error_percentage(real_third_value, predicted_third_value)
        fourth_error = calculate_error_percentage(real_fourth_value, predicted_fourth_value)
        
        high_errors.append(high_error)
        low_errors.append(low_error)
        third_errors.append(third_error)
        fourth_errors.append(fourth_error)
    
    # Calcul des statistiques
    avg_high_error = np.mean(high_errors)
    avg_low_error = np.mean(low_errors)
    avg_third_error = np.mean(third_errors)
    avg_fourth_error = np.mean(fourth_errors)

    print(f"Moyenne des erreurs sur les 'Highs' : {avg_high_error:.2f}%")
    print(f"Moyenne des erreurs sur les 'Lows' : {avg_low_error:.2f}%")
    print(f"Moyenne des erreurs sur la 3ème sortie : {avg_third_error:.2f}%")
    print(f"Moyenne des erreurs sur la 4ème sortie : {avg_fourth_error:.2f}%")
    
    plt.figure(figsize=(20, 6))

    # Graphique 1 : Comparaison Highs réels vs prédits
    plt.subplot(1, 4, 1)  # (Ligne, Colonne, Position)
    plt.plot(real_highs, label="High Réels", color="blue", marker="o")
    plt.plot(predicted_highs, label="High Prédits", color="orange", marker="x")
    plt.title("Comparaison des Highs (Réalité vs Prédictions)")
    plt.xlabel("Index des données")
    plt.ylabel("Valeur High")
    plt.legend()
    plt.grid()

    # Graphique 2 : Comparaison Lows réels vs prédits
    plt.subplot(1, 4, 2)
    plt.plot(real_lows, label="Low Réels", color="green", marker="o")
    plt.plot(predicted_lows, label="Low Prédits", color="red", marker="x")
    plt.title("Comparaison des Lows (Réalité vs Prédictions)")
    plt.xlabel("Index des données")
    plt.ylabel("Valeur Low")
    plt.legend()
    plt.grid()

    # Graphique 3 : Comparaison de la 3ème sortie réelle vs prédite
    plt.subplot(1, 4, 3)
    plt.plot(real_third, label="3ème Réel", color="purple", marker="o")
    plt.plot(predicted_third, label="3ème Prédit", color="brown", marker="x")
    plt.title("Comparaison de la 3ème Sortie (Réalité vs Prédictions)")
    plt.xlabel("Index des données")
    plt.ylabel("Valeur 3ème")
    plt.legend()
    plt.grid()

    # Graphique 4 : Comparaison de la 4ème sortie réelle vs prédite
    plt.subplot(1, 4, 4)
    plt.plot(real_fourth, label="4ème Réel", color="cyan", marker="o")
    plt.plot(predicted_fourth, label="4ème Prédit", color="magenta", marker="x")
    plt.title("Comparaison de la 4ème Sortie (Réalité vs Prédictions)")
    plt.xlabel("Index des données")
    plt.ylabel("Valeur 4ème")
    plt.legend()
    plt.grid()

    # Affichage des graphiques
    plt.tight_layout()  # Pour ajuster les espacements entre les graphiques
    plt.show()

if __name__ == "__main__":
    dataLen = int(input("dataLen {20}: "))
    system("clear"); system("tree data/")
    prepare_json_with_lookback(input_csv=f"data/{input("FileName: ")}", output_json="tmp/verification.json", lookback=dataLen)
    system('clear'); system('tree outputs/')
    evaluate_model(model_path=f"outputs/{input('FileName: ')}", json_file="tmp/verification.json", lookback=dataLen)