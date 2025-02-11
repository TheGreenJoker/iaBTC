# Import Libs
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json, os, datetime

# My Libs
from modele import SimpleModel
from create_data import prepare_json_with_lookback
from dataInit import CustomDataset

def main(epochs=100, json_file="data.json", model_path="model.pt", loss_path="losses.txt", lookback:int=10):
    dataset = CustomDataset(json_file)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = SimpleModel(lookback)

    criterion = nn.MSELoss()  # Erreur quadratique moyenne
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    losses = []

    for epoch in range(epochs):
        for batch_inputs, batch_targets in dataloader:
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        else:
            print(f"[{(epoch%10)*'##' + (10-(epoch%10))*'  '}]", end="\r")



    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    torch.save(model.state_dict(), f"outputs/{current_time}><{model_path}")
    print(f"Modèle sauvegardé dans : outputs/{current_time}><{model_path}")

    with open(f"outputs/{current_time}><{loss_path}", mode='w') as f:
        f.write("\n".join(map(str, losses)))
    print(f"Évolution des pertes sauvegardée dans : outputs/{current_time}><{loss_path}")

    test_input = torch.tensor([dataset.inputs[0].tolist()], dtype=torch.float32)
    predicted_output = model(test_input)
    print("Test Input:", test_input)
    print("Predicted Outputs:", predicted_output)

if __name__ == "__main__":
    dataLen = 30
    os.system("clear"); os.system("tree data/")
    prepare_json_with_lookback(input_csv=f"data/{input("FileName: ")}", output_json="tmp/data.json", lookback=dataLen)
    main(epochs=1000, json_file="tmp/data.json", model_path="model.pt", loss_path="losses.txt", lookback=dataLen)
