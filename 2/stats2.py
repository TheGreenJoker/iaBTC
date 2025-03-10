from os import system
import torch
import torch.nn as nn
from modele import get_lines
import matplotlib.pyplot as plt

global DEVICE 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(path):
    model = torch.load(path, map_location=DEVICE, weights_only=False)
    model.reset_memory(1, "cuda")
    model.eval()
    return model

def prompt(model, matrix):
    for vector in matrix:
        output = model(torch.tensor(vector, dtype=torch.float32).unsqueeze(0).to(DEVICE))
    return output.cpu().numpy()

def graphe(targets, outputs):
    if len(targets) != len(outputs):
        print("Les listes doivent avoir la même taille.")
        return

    t = range(len(targets))
    plt.plot(t, targets, label='Targets', color='blue')  # Première courbe en bleu
    plt.plot(t, outputs, label='Outputs', color='red')   # Deuxième courbe en rouge

    plt.xlabel('Temps (t)')
    plt.ylabel('Valeurs')
    plt.title('Graphique des deux courbes')
    
    plt.legend()
    plt.show()

def evaluate_modele(modele, matrice_2d_path, delta_t=32, prompt_nb=50):
    outputs = []
    targets = []

    for i in range(prompt_nb):
        inputs, target = get_lines(file=matrice_2d_path, line_cout=delta_t, line0=i)
        model.reset_memory(1, "cuda")
        
        for current_input in inputs:
            last_input = current_input
            current_input_sensor = torch.tensor(current_input, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            output = model(current_input_sensor).cpu().detach().numpy()
        output = output[0]

        outputs.append(output[0])
        targets.append(target[0])
        
        print(f"{output} <=> {target}")
    
    graphe(targets, outputs)

if __name__ == "__main__":
    system("tree modeles/")
    model_name = input("Filename: ")
    model_path = f"modeles/{model_name}"

    system("tree npy.tmp/")
    matrice_2d_path = f"npy.tmp/{input('Filename: ')}"

    model = get_model(model_path)

    prompt_nb = 50
    delta_t = 32

    evaluate_modele(model, matrice_2d_path, delta_t, prompt_nb)
