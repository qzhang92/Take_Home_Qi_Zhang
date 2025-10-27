from sklearn.cluster import DBSCAN
import numpy as np
from read import X
# from the previous step: X is the (N, EMBEDDING_DIM) feature matrix

# --- Configuration for DBSCAN ---
# These parameters are crucial and often require tuning based on given data.
# eps (epsilon): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
# min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
EPSILON = 0.5  # A starting value; adjust based on the density of code embeddings
MIN_SAMPLES = 5 # A reasonable minimum number of files to form a valid group

print("Start the training and clustering process")
print("1. Applying DBSCAN clustering to the feature matrix X...")

# Initialize and fit the DBSCAN model
db = DBSCAN(eps=EPSILON, min_samples=MIN_SAMPLES, metric='cosine').fit(X)

# Retrieve the cluster labels for each file
labels = db.labels_

# Determine the number of clusters found (excluding noise/outliers)
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_outliers_ = list(labels).count(-1)

print(f"DBSCAN clustering complete.")
print(f"  - Number of estimated clusters (groups): {n_clusters_}")
print(f"  - Number of estimated outliers (label -1): {n_outliers_}")

# Example output of labels (e.g., [0, 0, 1, -1, 0, 1, ...])
# print("\nFirst 10 cluster labels:", labels[:10])

# Initialize dictionaries to store centroids and group indices
cluster_centroids = {}
unique_labels = set(labels)

print("\n2. Calculating Centroids for each defined cluster...")

for k in unique_labels:
    if k == -1:
        # Skip the outlier/noise points
        continue
    
    # Get the indices of the data points belonging to cluster k
    class_member_mask = (labels == k)
    
    # Extract the feature vectors for the current cluster
    cluster_points = X[class_member_mask]
    
    # Calculate the centroid (mean vector) for this cluster
    centroid = np.mean(cluster_points, axis=0)
    
    # Store the centroid
    cluster_centroids[k] = centroid
    
    print(f"  - Centroid for Group {k} calculated (Size: {len(cluster_points)} files)")

# Example: Access the centroid of Group 0
# centroid_group_0 = cluster_centroids.get(0)
# print(f"\nCentroid of Group 0 (first 5 dimensions): {centroid_group_0[:5]}")

# cluster_centroids now holds the core identity vectors (C_k) for all valid groups.
# This structure is essential for the next step: classifying new files and checking deviation.
# Note: For the first time we run, we have no groups. We only have outliers. That is expected as no previous input is here.