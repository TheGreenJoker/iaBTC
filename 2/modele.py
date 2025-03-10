import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn import Module
from random import randint

# Définition du modèle de réseau de neurones
class ModeleLSTMnn(Module):
    def __init__(self, input_dim, output_dim):
        super(ModeleLSTMnn, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 32)
        self.lstm2 = torch.nn.LSTM(32, 128, batch_first=True)
        self.fc3 = torch.nn.Linear(128, 64)
        self.lstm4 = torch.nn.LSTM(64, 32, batch_first=True)
        self.fc5 = torch.nn.Linear(32, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # (batch, seq_len, 32)
        x, _ = self.lstm2(x)         # (batch, seq_len, 128)
        x = torch.relu(self.fc3(x))  # (batch, seq_len, 64)
        x, _ = self.lstm4(x)         # (batch, seq_len, 32)
        x = self.fc5(x)              # (batch, seq_len, output_dim)
        return x

    def reset_memory(self, batch_size, device):
        """ Réinitialiser la mémoire du modèle LSTM """
        # Réinitialiser h_0 et c_0 à zéro pour chaque LSTM
        self.h_0_lstm2 = torch.zeros(1, batch_size, self.lstm2.hidden_size).to(device)  # (num_layers, batch_size, hidden_size)
        self.c_0_lstm2 = torch.zeros(1, batch_size, self.lstm2.hidden_size).to(device)  # (num_layers, batch_size, hidden_size)
        
        self.h_0_lstm4 = torch.zeros(1, batch_size, self.lstm4.hidden_size).to(device)  # (num_layers, batch_size, hidden_size)
        self.c_0_lstm4 = torch.zeros(1, batch_size, self.lstm4.hidden_size).to(device)  # (num_layers, batch_size, hidden_size)
        
        return (self.h_0_lstm2, self.c_0_lstm2), (self.h_0_lstm4, self.c_0_lstm4)

def get_lines(file:str, line_cout:int, line0:int=False):
    matrix = np.load(file)
    totalT = len(matrix)

    line0 = line0 if (not line0) or (line0 == "random") else randint(0, len(matrix)-line_cout-1)

    if len(matrix) < (line0 + line_cout):
        print(f"[Warning] line0+line_count > matrix size => {line0 + line_cout}>{len(matrix)}, using random line0")
        line0 = randint(0, len(matrix)-line_cout-1) 
    inputValue = []
    for i in range(line_cout):
        inputValue.append(matrix[line0+i])
        outputValue = matrix[line0+i+1]
    
    return np.array(inputValue), np.array(outputValue)