import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Charger les données depuis le fichier CSV
data = pd.read_csv('dataTp2.csv')

# Sélectionner les caractéristiques émotionnelles pour le clustering
emotion_features = ['valence_intensity', 'fear_intensity', 'anger_intensity', 'happiness_intensity', 'sadness_intensity']
X = data[emotion_features]

# Appliquer le clustering hiérarchique avec 3 clusters
agg_clustering = AgglomerativeClustering(n_clusters=3)
predicted_labels = agg_clustering.fit_predict(X)

# Mapping des labels prédits aux sentiments réels
label_mapping = {0: 0, 1: 1, 2: -1}  # Ajuster selon vos résultats
mapped_labels = [label_mapping[label] for label in predicted_labels]

# Vérité terrain (sentiments réels)
true_labels = data['sentiment'].values

# Calculer les métriques d'évaluation
precision = precision_score(true_labels, mapped_labels, average='weighted')
recall = recall_score(true_labels, mapped_labels, average='weighted')
f1 = f1_score(true_labels, mapped_labels, average='weighted')

# Afficher les résultats
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-score: {f1:.4f}')
