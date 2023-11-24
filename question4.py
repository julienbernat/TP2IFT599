import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, precision_recall_fscore_support
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the data
data = pd.read_csv('datatp2.csv')

# Select the emotional features for clustering
emotion_features = ['valence_intensity', 'fear_intensity', 'anger_intensity', 'happiness_intensity', 'sadness_intensity']
X = data[emotion_features]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Separate the data by class (assuming you have class labels in the 'sentiment' column)
class_0 = X[data['sentiment'] == 0]
class_1 = X[data['sentiment'] == 1]
class_minus_1 = X[data['sentiment'] == -1]

# Concatenate the data from all classes
all_data = np.concatenate((class_0, class_1, class_minus_1))

# Apply K-means with K=3
kmeans_3 = KMeans(n_clusters=3, random_state=42)
kmeans_3.fit(all_data)

# Predict clusters for each sample
all_clusters_3 = kmeans_3.predict(all_data)

# Get the actual class labels
labels_class_0 = np.zeros(len(class_0))  # Labels for sentiment 0
labels_class_1 = np.ones(len(class_1))  # Labels for sentiment 1
labels_class_minus_1 = np.full(len(class_minus_1), -1)  # Labels for sentiment -1
all_labels = np.concatenate((labels_class_0, labels_class_1, labels_class_minus_1))

# Calculate silhouette score
silhouette_avg = silhouette_score(all_data, all_clusters_3)
print(f"Silhouette Score: {silhouette_avg}")

# Calculate precision, recall, and f1-score
precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_clusters_3, average='weighted')

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1_score}")
