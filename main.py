import pandas as pd
import cohere
import matplotlib.pyplot as plt

# Étape 1: Charger et préparer les données

# Charger le fichier CSV
df = pd.read_csv("store.csv")

# Fonction pour diviser les données en lots
def chunk_data(dataframe, chunk_size=100):
    # Diviser le dataframe en morceaux de taille spécifiée
    return [dataframe[i:i+chunk_size] for i in range(0, len(dataframe), chunk_size)]

# Diviser les données en lots de 100 lignes
chunks = chunk_data(df, chunk_size=100)

# Étape 2: Configurer l'API Cohere
co = cohere.Client("NSaRAXTmFQnar8mmKDD4qriFMKfx9lZ0UJSlQQpP")  # Remplacez par votre vraie clé API

# Étape 3: Envoi de chaque lot au LLM pour analyse

for idx, chunk in enumerate(chunks):
    # Convertir les données en texte (premières 5 lignes de chaque lot pour éviter trop de texte)
    data_text = chunk.head(5).to_string()
    
    response = co.generate(
        model="command-r-08-2024",  # Choisir le modèle approprié
        prompt=f"Voici les données des magasins (lot {idx+1}) :\n{data_text}\nAnalyse ces données et donne-moi des insights sur l'optimisation des ventes.",
        max_tokens=1000  # Limite le nombre de tokens générés
    )
    
    print(f"\nRéponse de l'IA pour le lot {idx+1}:")
    print(response.generations[0].text)

# Étape 4: Visualisation des données (optionnelle)

# Créer une visualisation des ventes en fonction de la surface du magasin
plt.figure(figsize=(8, 6))# Créer un graphique à dispersion
plt.scatter(df['Store_Area'], df['Store_Sales'], color='blue', label='Ventes par Surface')
plt.title('Ventes vs Surface du Magasin')
plt.xlabel('Surface du Magasin (m²)')
plt.ylabel('Ventes ($)')
plt.grid(True)
plt.legend()
plt.show()

# Étape 5: Prévisions des ventes (exemple basé sur les données)
# Exemple de prédiction basée sur les résultats générés
predicted_sales = response.generations[0].text
print("\nPrédiction des ventes pour un magasin de 1500 m² avec 300 articles :")
print(predicted_sales)
