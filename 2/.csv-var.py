import csv
from os import system
from datetime import datetime

# Afficher la structure du dossier data/
system("tree data/")

# Lire les données du fichier CSV
input_file = f"data/{input('File1: ')}"
output_file = f"data/{input('File2: ')}"

with open(input_file, newline="", encoding="utf-8") as fichier:
    lecteur = list(csv.reader(fichier))  # Convertir en liste pour un accès indexé
    donnees = lecteur[1:]  # Exclure l'en-tête

# Ouvrir un fichier pour écrire les variations
with open(output_file, "w", newline="", encoding="utf-8") as fichier_sortie:
    ecrivain = csv.writer(fichier_sortie)

    # Calculer les variations
    for i in range(1, len(donnees)):
        heure = datetime.strptime(donnees[i][1], "%Y-%m-%d %H:%M:%S").strftime("%H%M%S")
        ligne_precedente = list(map(float, [donnees[i - 1][3], donnees[i - 1][4], donnees[i - 1][5], donnees[i - 1][6], donnees[i - 1][7]]))
        ligne_actuelle = list(map(float, [donnees[i][3], donnees[i][4], donnees[i][5], donnees[i][6], donnees[i][7]]))
        variation = [((act - prec) / prec) * 100 if prec != 0 else 0 for prec, act in zip(ligne_precedente, ligne_actuelle)]
        ecrivain.writerow([heure] + variation)

print("Calcul des variations terminé. Résultats enregistrés dans", output_file)
