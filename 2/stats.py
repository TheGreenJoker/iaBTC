import torch
import matplotlib.pyplot as plt
from os import system
from modele import MatrixNN, load_data_from_npz

if __name__ == "__main__":
    # Charger la matrice de test
    system("tree npy.tmp/")
    file_path = f"npy.tmp/{input('FileName: ')}"
    x_test, y_test = load_data_from_npz(file_path)

    # Applatissement des matrices de test
    n_samples = x_test.shape[0]
    input_dim = x_test.shape[1] * x_test.shape[2]
    x_test = x_test.reshape(n_samples, -1)

    # Charger le modèle sauvegardé
    model = MatrixNN(input_dim, y_test.shape[1])
    system("tree modeles/")
    model.load_state_dict(torch.load(f"modeles/{input('FileName: ')}"))
    model.eval()

    # Effectuer des prédictions
    predictions = model(torch.tensor(x_test, dtype=torch.float32))

    # Calculer l'erreur en pour mille pour chaque échantillon
    absolute_error = torch.abs(predictions - torch.tensor(y_test, dtype=torch.float32))
    relative_error = absolute_error / torch.abs(torch.tensor(y_test, dtype=torch.float32))

    # Calculer l'erreur en pour mille
    error_per_mille = relative_error * 1000  # En milles

    # Calculer la moyenne de l'erreur en pour mille
    mean_error_per_mille = torch.mean(error_per_mille)

    # Calcul de l'erreur de signe (vérifier les signes)
    sign_error = torch.sign(predictions) != torch.sign(torch.tensor(y_test, dtype=torch.float32))
    
    # Calcul de la proportion d'erreur de signe en pour mille
    sign_error_per_mille = torch.mean(sign_error.float()) * 1000  # On le multiplie par 1000 pour obtenir en pour mille

    print(f"Erreur moyenne en pour mille sur le jeu de test : {mean_error_per_mille.item():.4f}")
    print(f"Proportion d'erreur de signe en pour mille sur le jeu de test : {sign_error_per_mille.item():.4f}")

    # Tracer les courbes des valeurs réelles vs. prédites
    plt.figure(figsize=(10, 6))

    # Tracer les valeurs réelles
    plt.plot(y_test[:10, :], label="Valeurs réelles", marker='o', linestyle='-', color='b')

    # Tracer les valeurs prédites
    plt.plot(predictions[:10, :].detach().numpy(), label="Valeurs prédites", marker='x', linestyle='--', color='r')

    plt.xlabel('Index des échantillons')
    plt.ylabel('Valeurs')
    plt.title('Comparaison des valeurs réelles et des valeurs prédites')

    # Ajouter la légende
    plt.legend()

    # Afficher la courbe
    plt.show()
