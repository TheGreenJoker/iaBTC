import torch, json
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, json_file):
        # Charger les données depuis le fichier JSON
        with open(json_file, 'r') as f:
            data = json.load(f)
        self.inputs = torch.tensor([item['input'] for item in data], dtype=torch.float32)
        self.targets = torch.tensor([item['output'] for item in data], dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]