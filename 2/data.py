import csv
from numpy import array, save
from os import system

def csv_to_matrix(csvFileName, npyFileName, lookback):
    with open(csvFileName, newline="", encoding="utf-8") as file:
        lecteur = csv.reader(file)

        buffer = []
        count = 0
        all_matrices = []  # Liste pour accumuler toutes les matrices
        
        for line in lecteur:
            # Convertir chaque élément de la ligne en float (ou int si les données sont entières)
            line = [float(x) for x in line]
            
            if len(buffer) < lookback:
                buffer.append(line)
            elif len(buffer) > lookback:
                buffer.pop(0)
            if len(buffer) != lookback:
                continue
            
            # Ajouter la matrice actuelle à la liste
            all_matrices.append(array(buffer))
            count += 1
        
        # Sauvegarder toutes les matrices sous forme d'un seul fichier .npy
        save(npyFileName.replace(".csv", ''), array(all_matrices))
            
if __name__ == "__main__":
    system("tree data/")
    name = input("Filename: ")
    csv_to_matrix(f"data/{name}", f"npy.tmp/{name}", lookback=8)
