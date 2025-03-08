import csv
from os import system
# Lire les données du fichier CSV

system("tree data/")
with open(f"data/{input("File1: ")}", newline="", encoding="utf-8") as fichier:
    lecteur = list(csv.reader(fichier))  # Convertir en liste pour un accès indexé
    
    en_tete = lecteur[0][2:]  # Extraire l'en-tête en supprimant les deux premières colonnes
    donnees = lecteur[1:]  # Exclure l'en-tête

# Ouvrir un fichier pour écrire les variations
with open(f"data/{input("File2: ")}", "w", newline="", encoding="utf-8") as fichier_sortie:
    ecrivain = csv.writer(fichier_sortie)
    
    # Calculer les variations
    for i in range(1, len(donnees)):
        ligne_precedente = list(map(float, donnees[i - 1][2:]))  # Convertir en float
        ligne_actuelle = list(map(float, donnees[i][2:]))
        variation = [((act - prec) / prec) * 100 for prec, act in zip(ligne_precedente, ligne_actuelle)]
        
        # Écrire la ligne avec les variations
        ecrivain.writerow(variation)

print("Calcul des variations terminé. Résultats enregistrés dans variations.csv")
