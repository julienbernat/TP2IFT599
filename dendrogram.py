import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler

# Charger les données depuis le fichier CSV
data = pd.read_csv('dataTp2.csv')

# Créer un dossier pour sauvegarder les figures s'il n'existe pas déjà
output_folder = 'figures'
os.makedirs(output_folder, exist_ok=True)

# Sélectionner les caractéristiques émotionnelles pour le clustering
emotion_features = ['valence_intensity', 'fear_intensity', 'anger_intensity', 'happiness_intensity', 'sadness_intensity']
X = data[emotion_features]

# Standardiser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Configurer le style de seaborn pour de meilleures visualisations
sns.set(style="whitegrid")

# Valeurs de seuil à tester
thresholds = [10, 15, 20, 25]  # Ajoutez d'autres valeurs de seuil si nécessaire

# Créer plusieurs sous-plots pour les dendrogrammes
fig, axes = plt.subplots(nrows=len(thresholds), ncols=1, figsize=(15, 8 * len(thresholds)))

# Générer les dendrogrammes pour chaque valeur de seuil
for i, threshold in enumerate(thresholds):
    linked = linkage(X_scaled, 'ward')  # Vous pouvez ajuster la méthode de lien ici
    clusters = fcluster(linked, t=threshold, criterion='distance')

    dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True, ax=axes[i])
    axes[i].set_title(f'Dendrogramme Hiérarchique - Seuil : {threshold}')
    axes[i].set_xlabel('Indice de l\'échantillon')
    axes[i].set_ylabel('Distance')

    # Afficher les clusters obtenus à partir du seuil
    axes[i].axhline(y=threshold, color='r', linestyle='--', label=f'Seuil : {threshold}')
    axes[i].legend()

# Ajuster l'espacement entre les sous-plots
plt.tight_layout()

# Nommer et sauvegarder la figure dans le dossier spécifié
figure_name = 'dendrograms.png'
figure_path = os.path.join(output_folder, figure_name)
plt.savefig(figure_path)

plt.show()
