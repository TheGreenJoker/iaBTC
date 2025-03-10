import csv
from numpy import array, save
from os import system

def csv_to_matrix(csvFileName, npyFileName, lookback):
    with open(csvFileName, newline="", encoding="utf-8") as file:
        lecteur = csv.reader(file)
        matrices = []
        
        for line in lecteur:
            # Convertir chaque élément de la ligne en float (ou int si les données sont entières)
            line = [float(x) for x in line]
            matrices.append(array(line))
        
        save(npyFileName.replace(".csv", ''), array(matrices))
            
if __name__ == "__main__":
    system("tree data/")
    name = input("Filename: ")
    csv_to_matrix(f"data/{name}", f"npy.tmp/{name}", lookback=8)
