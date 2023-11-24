import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données depuis le fichier CSV
data = pd.read_csv('dataTp2.csv')

# Liste des caractéristiques émotionnelles
emotion_features = ['valence_intensity', 'fear_intensity', 'anger_intensity', 'happiness_intensity', 'sadness_intensity']

# Configurer le style de seaborn pour de meilleures visualisations
sns.set(style="whitegrid")

# Créer un dossier pour sauvegarder les figures s'il n'existe pas déjà
output_folder = 'figures'
os.makedirs(output_folder, exist_ok=True)

# Créer les figures avec scatter plot et histogramme superposé
for i in range(len(emotion_features)):
    for j in range(i+1, len(emotion_features)):
        plt.figure(figsize=(12, 6))
        
        # Scatter plot pour visualiser la distribution conjointe
        sns.scatterplot(x=emotion_features[i], y=emotion_features[j], data=data, alpha=0.7)
        
        # Histogramme pour la première caractéristique
        sns.histplot(data[emotion_features[i]], kde=True, color='blue', label=emotion_features[i], alpha=0.5)
        
        # Histogramme pour la deuxième caractéristique
        sns.histplot(data[emotion_features[j]], kde=True, color='orange', label=emotion_features[j], alpha=0.5)
        
        # Ajouter des labels, un titre et une légende
        plt.xlabel(emotion_features[i])
        plt.ylabel(emotion_features[j])
        plt.title(f'Distribution conjointe de {emotion_features[i]} et {emotion_features[j]} avec histogrammes')
        plt.legend()
        
        # Nommer et sauvegarder la figure dans le dossier spécifié
        figure_name = f'{emotion_features[i]}_{emotion_features[j]}_distribution.png'
        figure_path = os.path.join(output_folder, figure_name)
        plt.savefig(figure_path)
        
        # Afficher la figure
        plt.show()

        print(f'Figure enregistrée sous le chemin : {figure_path}')
