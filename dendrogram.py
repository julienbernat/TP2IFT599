import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

# Charger les données depuis le fichier CSV
data = pd.read_csv('dataTp2.csv')

# Sélectionner les caractéristiques émotionnelles pour le clustering
emotion_features = ['valence_intensity', 'fear_intensity', 'anger_intensity', 'happiness_intensity', 'sadness_intensity']
X = data[emotion_features]

# Configurer le style de seaborn pour de meilleures visualisations
sns.set(style="whitegrid")

# Appliquer la méthode de linkage hiérarchique
linked = linkage(X, 'ward')

# Afficher le dendrogramme
plt.figure(figsize=(12, 8))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Dendrogramme Hiérarchique')
plt.xlabel('Indice de l\'échantillon')
plt.ylabel('Distance')
plt.show()
