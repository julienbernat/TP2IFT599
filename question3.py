import pandas as pd
import scipy
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

def overlap(class1, class2, distance_function, mahalanobis=False, cov_matrix=None):
    def intra_class_distance(data, mahalonobis=False, cov_matrix=None):
        centroid_class = np.mean(data, axis=0)
        intra_distance = 0
        for i in range(len(data)):
            if mahalonobis:
                distance = distance_function(data[i], centroid_class, cov_matrix)
            else:
                distance = distance_function(data[i], centroid_class)
            if distance > intra_distance:
                intra_distance = distance
        return intra_distance

    def class_distance(class1, class2, distance_function, mahalonobis=False, cov_matrix=None):
        class_distance = 1000000000
        centroid_class2 = np.mean(class2, axis=0)
        for i in range(len(class1)):
            if mahalonobis:
                distance = distance_function(class1[i], centroid_class2, cov_matrix)
            else:
                distance = distance_function(class1[i], centroid_class2)
            if distance < class_distance:
                class_distance = distance
        return class_distance

    def inter_class_distance(class1, class2, distance_function, mahalonobis=False, cov_matrix=None):
        distance1 = class_distance(class1, class2, distance_function, mahalonobis, cov_matrix)
        distance2 = class_distance(class2, class1, distance_function, mahalonobis, cov_matrix)
        if distance1 < distance2:
            return distance1
        else:
            return distance2

    overlap_value = (
        intra_class_distance(class1, mahalanobis, cov_matrix)
        + intra_class_distance(class2, mahalanobis, cov_matrix)
    ) / (2 * inter_class_distance(class1, class2, distance_function, mahalanobis, cov_matrix))

    return overlap_value

# Initialize an empty list to store the results
results_list = []

# Iterate over different values of K
for k in range(2, 11):
    # Apply K-means
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)

    # Predict clusters for each sample
    clusters = kmeans.predict(X_scaled)

    # Calculate silhouette score
    silhouette_avg = silhouette_score(X_scaled, clusters)

    # Calculate the overlap using Mahalanobis distance
    overlap_value = overlap(X_scaled[clusters == 0], X_scaled[clusters == 1], scipy.spatial.distance.mahalanobis, True, np.cov(X_scaled, rowvar=False))

    if (k==3):
        actual_labels = data['sentiment'].values
        precision, recall, f1_score, _ = precision_recall_fscore_support(actual_labels, clusters, average='weighted')

    # Append the results to the list
    results_list.append({
        'K': k,
        'Silhouette Score': silhouette_avg,
        'Overlap': overlap_value,
    })

# Create a DataFrame from the list of results
results_table = pd.DataFrame(results_list)

# Print the results table
print(results_table)

print('Precision: ',precision, 'Recall: ',recall, 'F1_score: ',f1_score)
