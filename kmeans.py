import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from umap import UMAP

# Charger les données depuis le fichier CSV
data = pd.read_csv('dataTp2.csv')

# Sélectionner les caractéristiques émotionnelles pour le clustering
emotion_features = ['valence_intensity', 'fear_intensity', 'anger_intensity', 'happiness_intensity', 'sadness_intensity']
X = data[emotion_features]

# Liste des valeurs de K à tester
k_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]

# Créer un dossier pour sauvegarder les figures s'il n'existe pas déjà
output_folder = 'figures'
os.makedirs(output_folder, exist_ok=True)

# Configurer le style de seaborn pour de meilleures visualisations
sns.set(style="whitegrid")

# Itérer sur les différentes valeurs de K
for i, k in enumerate(k_values):
    # Appliquer K-means
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)

    # Appliquer UMAP pour la réduction de dimension
    umap_model = UMAP(n_components=2, random_state=42)
    embedding = umap_model.fit_transform(X)

    # Visualiser le nuage de points avec coloration par cluster
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=labels, palette='viridis')
    plt.title(f'K-means (K={k}) - UMAP Projection')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')

    # Nommer et sauvegarder la figure dans le dossier spécifié
    figure_name = f'kmeans_umap_projection_k_{k}.png'
    figure_path = os.path.join(output_folder, figure_name)
    plt.savefig(figure_path)

    # Afficher et fermer la figure pour libérer la mémoire
    plt.show()
    plt.close()

    print(f'Figure enregistrée sous le chemin : {figure_path}')
