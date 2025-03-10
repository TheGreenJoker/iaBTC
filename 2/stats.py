from os import system
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Définition du modèle LSTM (doit être identique à celui utilisé lors de l'entraînement)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x, hidden_state=None):
        # x : [batch_size, seq_length, input_size]
        out, (h_n, c_n) = self.lstm(x, hidden_state)
        # On ne retient que la dernière sortie de la séquence
        out = self.fc(out[:, -1, :])
        return out, (h_n, c_n)

# Configuration de l'appareil de calcul
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chargement du modèle sauvegardé
system("tree modeles/")
model_path = f"modeles/{input('File: ')}"
model = torch.load(model_path, map_location=device)
model.to(device)
model.eval()

# Chargement des données de validation (fichier NumPy)
system("tree npy.tmp/")
val_filename = f"npy.tmp/{input('File: ')}"
validation_data = np.load(val_filename)
cutyn = input("Cut: ")
if cutyn:
    validation_data = validation_data[:int(cutyn)]

# Choix de la taille de la fenêtre (delta_t) pour les séquences d'entrée
delta_t = int(input("Entrer le delta_t (ex: 10): "))

# Préparation des entrées et cibles à partir des données de validation
inputs = []
targets = []
N = validation_data.shape[0]

for i in range(N - delta_t):
    # La séquence d'entrée est une tranche de delta_t observations
    seq_input = validation_data[i:i+delta_t]
    # La cible est la valeur (par exemple, la première feature) suivant la séquence
    seq_target = validation_data[i + delta_t, 0]
    inputs.append(torch.tensor(seq_input, dtype=torch.float32))
    targets.append(seq_target)

inputs = torch.stack(inputs).to(device)

# Si les données ont une dimension supplémentaire (ex: [num_samples, delta_t, 1, features]),
# on la retire pour obtenir la forme attendue par le LSTM: [batch, seq_length, features].
if inputs.dim() == 4:
    inputs = inputs.squeeze(2)

targets = np.array(targets)

# Passage en mode évaluation (aucune rétropropagation)
with torch.no_grad():
    predictions, _ = model(inputs)
    predictions = predictions.cpu().numpy().flatten()

# Tracé des courbes de prédictions et de valeurs réelles
plt.figure(figsize=(12, 6))
plt.plot(predictions, label="Prédictions", marker='o')
plt.plot(targets, label="Valeurs réelles", marker='x')
plt.title("Comparaison des Prédictions et des Valeurs Réelles")
plt.xlabel("Index")
plt.ylabel("Valeur")
plt.legend()
plt.show()
