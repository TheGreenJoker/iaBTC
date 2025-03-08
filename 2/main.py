from os import system
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from modele import MatrixNN, MatrixDataset, load_data_from_npz

if __name__ == "__main__":
    # Charger les données depuis "npy.tmp/matrix.npy"
    system("tree npy.tmp/")
    file_path = f"npy.tmp/{input("Filename: ")}"
    model_path = f"modeles/{input("Output: ")}.pth"
    x_train, y_train = load_data_from_npz(file_path)

    # Dimension des matrices (aplanissement des matrices 2D)
    n_samples = x_train.shape[0]
    input_dim = x_train.shape[1] * x_train.shape[2]  # Aplatir chaque matrice
    output_dim = y_train.shape[1]  # La dimension du vecteur de sortie

    # Aplatir les matrices 2D en vecteurs 1D
    x_train = x_train.reshape(n_samples, -1)

    # Créer le DataLoader pour les lots d"entraînement
    dataset = MatrixDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialiser le modèle
    model = MatrixNN(input_dim, output_dim)

    # Fonction de perte et optimiseur
    criterion = nn.MSELoss()  # Pour la régression
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Entraîner le modèle
    epochs = 50
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for i, (inputs, targets) in enumerate(dataloader):
            # Passage avant
            outputs = model(inputs)
            
            # Calculer la perte
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    # Sauvegarder le modèle
    torch.save(model.state_dict(), model_path)
    print(f"Modèle sauvegardé sous {model_path}")
