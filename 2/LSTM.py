from os import system
import numpy as np

import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from modele import ModeleLSTMnn, get_lines


# GPU/CPU
global DEVICE 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def train_model(matrice_2d_path, epochs_dict, model_name="mnist_experiment"):
    data_len, data_size = np.load(matrice_2d_path).shape

    model = ModeleLSTMnn(data_size, data_size).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Initialisation de TensorBoard
    writer = SummaryWriter(f"runs/{model_name}")

    stop = False
    for delta_t, epochs in epochs_dict.items():
        if stop:
            continue
        delta_t = int(delta_t)
        for epoch in range(epochs):
            
            # == Réinitialiser la mémoire à chaque cycle == #
            optimizer.zero_grad()
            model.train()
            model.reset_memory(1, "cuda")
            # ============================================= #

            inputs, target = get_lines(file=matrice_2d_path, line_cout=delta_t)
            target_tensor = torch.tensor(target, dtype=torch.float32).to(DEVICE)

            for current_input in inputs:
                current_input_sensor = torch.tensor(current_input, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                output = model(current_input_sensor)

            loss = criterion(output.squeeze(0), target_tensor)

            loss.backward()
            optimizer.step()
            
            print(f"Δt={delta_t}, Epoch={epoch}/{epochs}, Loss={loss.item()}")
            writer.add_scalar(f"Loss/delta_t_{delta_t}", loss.item(), epoch)

    input("Press enter...")
    writer.close()

    return model


if __name__ == "__main__":
    # Charger les données depuis "npy.tmp/matrix.npy"
    system("tree npy.tmp/")
    matrice_2d_path = f"npy.tmp/{input('Filename: ')}"
    model_name = input("Output: ")
    model_path = f"modeles/{model_name}.pth"

    epochs_dict = {
        "1":2000,
        "2":1000,
        "3":1000,
        "4":1000,
        "8":1000,
        "12":3000,
        "16":1000,
        "24":4000,
        "32":1000
    }

    model = train_model(matrice_2d_path, epochs_dict, model_name)
    torch.save(model, model_path)
    print(f"Model saved at {model_path}.")
