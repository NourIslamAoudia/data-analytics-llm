# Analyseur de Données de Magasin

## Description
Ce projet est un outil d'analyse avancée des données de magasin qui combine l'analyse de données traditionnelle avec l'intelligence artificielle via l'API Cohere. Il permet d'analyser les performances des magasins, de visualiser les tendances et de faire des prédictions de ventes.

## Fonctionnalités
- 📊 Analyse de données par lots avec l'API Cohere
- 📈 Visualisations avancées des données
- 🤖 Modèle de prédiction des ventes
- 📑 Génération de rapports d'analyse
- 🧮 Calcul de métriques de performance

## Prérequis
- Python 3.8 ou supérieur
- Une clé API Cohere valide

## Installation

1. Clonez le repository :
```bash
git clone https://github.com/votre-username/store-analysis.git
cd store-analysis
```

2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

## Configuration
1. Créez un compte sur [Cohere](https://cohere.ai/)
2. Obtenez votre clé API
3. Remplacez `VOTRE_CLE_API_COHERE` dans le code par votre clé API

## Structure des données
Le fichier CSV d'entrée doit contenir au minimum les colonnes suivantes :
- `Store_Sales` : Ventes du magasin
- `Items_Available` : Nombre d'articles disponibles

## Utilisation

```python
from store_analyzer import StoreAnalyzer

# Initialisation
analyzer = StoreAnalyzer("VOTRE_CLE_API_COHERE")

# Chargement des données
df = analyzer.load_data("store.csv")

# Analyse des données
analyses = analyzer.analyze_chunks()

# Création des visualisations
analyzer.create_visualizations()

# Prédiction des ventes
prediction = analyzer.predict_sales(1500)
```

## Exemples de Visualisations
L'outil génère plusieurs types de visualisations :
- Graphique de dispersion des ventes
- Distribution des ventes
- Box plot des performances
- Évolution temporelle des ventes

## Métriques de Performance
Le modèle de prédiction fournit les métriques suivantes :
- R² Score
- RMSE (Root Mean Square Error)
- Coefficients de régression

## Structure du Projet
```
store-analysis/
│
├── store_analyzer.py     # Code principal
├── requirements.txt      # Dépendances
├── README.md            # Documentation
└── examples/            # Exemples d'utilisation
```

## Contribution
Les contributions sont les bienvenues ! N'hésitez pas à :
1. Fork le projet
2. Créer une branche (`git checkout -b feature/AmazingFeature`)
3. Commit vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Push sur la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request



## Remerciements
- Cohere pour leur API d'analyse de texte
- La communauté Python pour les bibliothèques utilisées
- Tous les contributeurs du projet
