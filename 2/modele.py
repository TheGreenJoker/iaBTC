import torch
import numpy as np

# Définition du modèle de réseau de neurones
class MatrixNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MatrixNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 32)
        self.fc2 = torch.nn.Linear(32, 128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.fc4 = torch.nn.Linear(64, 32)
        self.fc5 = torch.nn.Linear(32, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x


# Fonction pour charger les données depuis "npy.tmp/matrix.npy"
def load_data_from_npz(file_path):
    matrices_3d = np.load(file_path)

    matrices = []
    targets = []
    
    if matrices_3d.ndim != 3:
        raise ValueError("Le fichier doit contenir une matrice 3D.")
    
    for matrix in matrices_3d:
        if matrix.ndim != 2:
            continue
        
        matrix = matrix.astype(np.float32)
        target = matrix[-1, :]
        matrix_without_last_row = matrix[:-1, :]
        
        matrices.append(matrix_without_last_row)
        targets.append(target)
    
    return np.array(matrices), np.array(targets)