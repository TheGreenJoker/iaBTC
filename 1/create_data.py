import csv
import json

def prepare_json_with_lookback(input_csv, output_json, lookback=10):
    """
    Préparer un fichier JSON où chaque entrée contient les données des `lookback` jours précédents.
    """
    data = []
    with open(input_csv, mode='r', encoding='utf-8') as csv_file:
        csv_reader = list(csv.reader(csv_file, delimiter=';'))

        # Vérifiez qu'il y a assez de lignes pour créer des entrées
        for i in range(lookback, len(csv_reader) - 1):  # Ignorer les premières lignes pour lookback
            inputs = []
            for j in range(i - lookback, i):
                row = csv_reader[j]
                inputs.extend([
                    float(row[2]),  # Open
                    float(row[3]),  # High
                    float(row[4]),  # Low
                    float(row[5])   # Close
                ])
            # Les colonnes Low et High de la ligne suivante pour output
            next_row = csv_reader[i + 1]
            output_target = [float(next_row[2]), float(next_row[4]), float(next_row[3]), float(next_row[5])]  # Low et High du jour suivant
            data.append({"input": inputs, "output": output_target})

    # Sauvegarder les données dans le fichier JSON
    with open(output_json, mode='w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4)
    print(f"Fichier JSON avec lookback {lookback} jours généré : {output_json}")

if __name__ == "__main__":
    input_csv_file = "data.csv"  
    output_json_file = "data.json"  
    prepare_json_from_csv(input_csv_file, output_json_file)
