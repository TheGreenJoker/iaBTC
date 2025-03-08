# Import Libs
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json, os, datetime
import threading
import matplotlib.pyplot as plt
import numpy as np

# My Libs
from modele import SimpleModel
from create_data import prepare_json_with_lookback
from dataInit import CustomDataset

# Détection du GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Variable globale pour l'arrêt manuel
stop_training = False

def listen_for_keypress():
    """Arrête l'entraînement si 'Enter' est pressé"""
    global stop_training
    input("\n🔴 Appuyez sur 'Enter' pour arrêter l'entraînement...\n")
    stop_training = True
    print("\n🛑 Arrêt manuel demandé. Sauvegarde du modèle...")

def plot_loss_and_lr(losses, lrs):
    """Affiche la courbe de loss et de learning rate en temps réel"""
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(losses, label="Loss", color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(lrs, label="Learning Rate", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("LR")
    plt.title("Learning Rate Evolution")
    plt.legend()

    plt.pause(0.1)

def main(epochs=100, json_file="data.json", model_path="model.pt", loss_path="losses.txt", lr_path="lrs.txt", lookback=10):
    global stop_training

    dataset = CustomDataset(json_file)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = SimpleModel(lookback).to(device)  # Envoi du modèle sur GPU
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    losses = []
    lrs = []

    patience = 10
    tolerance = 1e-2
    worse_epochs = 0
    best_loss = float('inf')

    plt.ion()  # Mode interactif pour Matplotlib

    for epoch in range(epochs):
        if stop_training:
            break  # Arrêt immédiat si "Enter" a été pressé

        for batch_inputs, batch_targets in dataloader:
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        current_loss = loss.item()
        losses.append(current_loss)

        # 🔥 Récupérer et stocker le learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        lrs.append(current_lr)

        # Mise à jour du scheduler
        scheduler.step(current_loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {current_loss:.4f}, LR: {current_lr:.6f}")
        else:
            print(f"[{(epoch%10)*'##' + (10-(epoch%10))*'  '}] Loss: {current_loss:.4f}, LR: {current_lr:.6f}", end="\r")

        plot_loss_and_lr(losses, lrs)  # Mise à jour des courbes

    plt.ioff()  # Désactiver le mode interactif

    # Sauvegarde du modèle et des pertes
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), f"outputs/{current_time}--{model_path}")
    print(f"\n💾 Modèle sauvegardé dans : outputs/{current_time}--{model_path}")

    with open(f"outputs/{current_time}--{loss_path}", mode='w') as f:
        f.write("\n".join(map(str, losses)))
    print(f"📊 Évolution des pertes sauvegardée dans : outputs/{current_time}--{loss_path}")

    with open(f"outputs/{current_time}--{lr_path}", mode='w') as f:
        f.write("\n".join(map(str, lrs)))
    print(f"📉 Évolution du learning rate sauvegardée dans : outputs/{current_time}--{lr_path}")

    # Test avec un exemple d'entrée
    test_input = torch.tensor([dataset.inputs[0].tolist()], dtype=torch.float32).to(device)
    predicted_output = model(test_input)
    print("Test Input:", test_input.cpu())
    print("Predicted Outputs:", predicted_output.cpu())

if __name__ == "__main__":
    dataLen = int(input(r"LookBack {20}: "))
    epochs = int(input(r"Epochs {100}: "))
    
    os.system("clear")
    os.system("tree data/")

    prepare_json_with_lookback(input_csv=f"data/{input("FileName: ")}", output_json="tmp/data.json", lookback=dataLen)

    # ✅ Lancer le thread SEULEMENT après avoir pris les entrées utilisateur
    threading.Thread(target=listen_for_keypress, daemon=True).start()

    main(epochs=epochs, json_file="tmp/data.json", model_path="model.pt", loss_path="losses.txt", lr_path="lrs.txt", lookback=dataLen)
