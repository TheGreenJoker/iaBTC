from os import system
import pandas as pd

# Charger le fichier CSV1
system("tree ./")
df = pd.read_csv(input("FileName: "))

# Convertir les dates dans le bon format
df['End'] = pd.to_datetime(df['End'])
df['Start'] = pd.to_datetime(df['Start'])

# Liste pour stocker les nouvelles lignes
output_lines = []

# Calculer les variations entre les lignes successives
for i in range(1, len(df)):
    date_str = df['Start'].iloc[i].strftime("%d/%m/%y")
    
    # Calculer les variations
    open_var = ((df['Open'].iloc[i] - df['Open'].iloc[i-1]) / df['Open'].iloc[i-1]) * 100
    high_var = ((df['High'].iloc[i] - df['High'].iloc[i-1]) / df['High'].iloc[i-1]) * 100
    low_var = ((df['Low'].iloc[i] - df['Low'].iloc[i-1]) / df['Low'].iloc[i-1]) * 100
    close_var = ((df['Close'].iloc[i] - df['Close'].iloc[i-1]) / df['Close'].iloc[i-1]) * 100
    volume_var = ((df['Volume'].iloc[i] - df['Volume'].iloc[i-1]) / df['Volume'].iloc[i-1]) * 100
    market_cap_var = ((df['Market Cap'].iloc[i] - df['Market Cap'].iloc[i-1]) / df['Market Cap'].iloc[i-1]) * 100

    # Ajouter la ligne formatée dans la liste
    line = f"BTC000000000;{date_str};{open_var:.3f};{high_var:.3f};{low_var:.3f};{close_var:.3f};{volume_var:.3f};{market_cap_var:.3f}"
    output_lines.append(line)

# Sauvegarder le fichier de sortie
with open("output.csv", "w") as f:
    f.write("\n".join(output_lines))

print("Transformation terminée avec succès!")
