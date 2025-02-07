import pandas as pd
import cohere
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

class StoreAnalyzer:
    def __init__(self, api_key):
        """
        Initialise l'analyseur de données de magasin
        :param api_key: Clé API Cohere
        """
        self.co = cohere.Client(api_key)
        self.df = None
        self.model = None

    def load_data(self, file_path):
        """
        Charge et prépare les données depuis un fichier CSV
        :param file_path: Chemin vers le fichier CSV
        """
        self.df = pd.read_csv(file_path)
        
        # Nettoyage des données
        self.df = self.df.dropna()  # Supprime les lignes avec des valeurs manquantes
        
        # Ajout de métriques calculées
        self.df['Sales_per_Item'] = self.df['Store_Sales'] / self.df['Items_Available']
        return self.df

    def analyze_chunks(self, chunk_size=100):
        """
        Analyse les données par lots avec Cohere
        :param chunk_size: Taille de chaque lot
        :return: Liste des analyses
        """
        analyses = []
        chunks = [self.df[i:i+chunk_size] for i in range(0, len(self.df), chunk_size)]

        for idx, chunk in enumerate(chunks):
            # Préparation des données pour l'analyse
            summary_stats = {
                'moyenne_ventes': chunk['Store_Sales'].mean(),
                'total_items': chunk['Items_Available'].sum(),
                'performance': chunk['Sales_per_Item'].mean()
            }

            # Création du prompt pour l'analyse
            prompt = f"""
            Analyse du lot {idx+1}:
            Moyenne des ventes: {summary_stats['moyenne_ventes']:.2f}
            Total des articles: {summary_stats['total_items']}
            Performance moyenne: {summary_stats['performance']:.2f}
            
            Fournir une analyse détaillée de ces métriques et des recommandations.
            """

            # Appel à l'API Cohere
            response = self.co.generate(
                model="command-r-08-2024",
                prompt=prompt,
                max_tokens=1000,
                temperature=0.7
            )
            
            analyses.append(response.generations[0].text)
        
        return analyses

    def create_visualizations(self):
        """
        Crée des visualisations avancées des données
        """
        # Configuration du style
        plt.style.use('seaborn')
        
        # Création d'une figure avec plusieurs sous-graphiques
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Graphique de dispersion
        plt.subplot(2, 2, 1)
        sns.scatterplot(data=self.df, x='Items_Available', y='Store_Sales')
        plt.title('Ventes vs Articles Disponibles')
        
        # 2. Distribution des ventes
        plt.subplot(2, 2, 2)
        sns.histplot(self.df['Store_Sales'], bins=30)
        plt.title('Distribution des Ventes')
        
        # 3. Box plot des ventes par catégorie
        plt.subplot(2, 2, 3)
        sns.boxplot(data=self.df, y='Store_Sales')
        plt.title('Distribution des Ventes (Box Plot)')
        
        # 4. Evolution des ventes (si données temporelles disponibles)
        plt.subplot(2, 2, 4)
        sns.lineplot(data=self.df, x=self.df.index, y='Store_Sales')
        plt.title('Evolution des Ventes')
        
        plt.tight_layout()
        plt.show()

    def train_prediction_model(self):
        """
        Entraîne un modèle de prédiction des ventes
        """
        # Préparation des features
        X = self.df[['Items_Available']]
        y = self.df['Store_Sales']

        # Division des données
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Entraînement du modèle
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

        # Évaluation du modèle
        y_pred = self.model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        return {
            'r2_score': r2,
            'rmse': rmse,
            'coefficient': self.model.coef_[0],
            'intercept': self.model.intercept_
        }

    def predict_sales(self, items_available):
        """
        Prédit les ventes pour un nombre d'articles donné
        :param items_available: Nombre d'articles disponibles
        :return: Prédiction des ventes
        """
        if self.model is None:
            self.train_prediction_model()
        
        prediction = self.model.predict([[items_available]])
        return prediction[0]

# Exemple d'utilisation
if __name__ == "__main__":
    # Initialisation
    analyzer = StoreAnalyzer("NSaRAXTmFQnar8mmKDD4qriFMKfx9lZ0UJSlQQpP")
    
    # Chargement des données
    df = analyzer.load_data("store.csv")
    
    # Analyse des données
    analyses = analyzer.analyze_chunks()
    
    # Création des visualisations
    analyzer.create_visualizations()
    
    # Entraînement du modèle et prédictions
    model_metrics = analyzer.train_prediction_model()
    prediction = analyzer.predict_sales(1500)
    
    print(f"Prédiction des ventes pour 1500 articles: {prediction:.2f}$")
    print(f"Métriques du modèle: {model_metrics}")