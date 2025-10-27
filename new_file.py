import numpy as np
from read import X
from train_cluster import labels, cluster_centroids
from scipy.spatial.distance import cosine
from typing import Dict
# Assume all prerequisite variables (model, custom_tokenizer, generate_document_vector, EMBEDDING_DIM) are available.
'''
# --- Simulation of New File Content ---
# In a real environment, you would read the content of '101.txt'.
SIMULATED_CONTENT_101 = "PUSH EAX; ADD EBX, 4; LOOP LABEL_A; SUB EAX, 1" 

# NOTE: The custom_tokenizer and generate_document_vector functions are assumed here.
def custom_tokenizer(content):
    # Placeholder for the function from Step 2
    return content.upper().replace(';', ' ').replace(',', ' ').split() 

def generate_document_vector(tokens, model):
    # Placeholder for the function from Step 2
    valid_vectors = [model.wv[t] for t in tokens if t in model.wv]
    return np.mean(valid_vectors, axis=0) if valid_vectors else np.zeros(100)

# Execute Step 1
tokens_101 = custom_tokenizer(SIMULATED_CONTENT_101)
V_new = generate_document_vector(tokens_101, model)
print("1. New File Encoding Complete.")
print(f"   V_new vector generated. Shape: {V_new.shape}")
'''
# Assume X, labels, and cluster_centroids are available from Step 3.
# Also assume the necessary functions (calculate_threshold, Z_FACTOR) are defined.

def calculate_threshold(X: np.ndarray, labels: np.ndarray, centroids: Dict[int, np.ndarray], z_factor: float = 2.0) -> float:
    # Function body from the previous response (calculates μ + Zσ)
    intra_cluster_distances = []
    for k in centroids.keys():
        distances = [cosine(point, centroids[k]) for point in X[labels == k]]
        intra_cluster_distances.extend(distances)
    if not intra_cluster_distances: return float('inf') 
    mu, sigma = np.mean(intra_cluster_distances), np.std(intra_cluster_distances)
    return mu + z_factor * sigma

# --- Execute Step 2a: Calculate Threshold (Theta) ---
Z_FACTOR = 2.0 
THRESHOLD = calculate_threshold(X, labels, cluster_centroids, Z_FACTOR)
print("\n2. Distance and Threshold Check:")
print(f"   2a. Anomaly Threshold (Theta, μ + {Z_FACTOR}σ): {THRESHOLD:.4f}")

# --- Execute Step 2b: Calculate Minimum Distance (D_min) ---
min_distance = float('inf')
closest_cluster = -1 

for k, centroid in cluster_centroids.items():
    distance = cosine(V_new, centroid)
    
    if distance < min_distance:
        min_distance = distance
        closest_cluster = k

print(f"   2b. Minimum Distance (D_min) to closest centroid: {min_distance:.4f}")

# Assume min_distance, closest_cluster, and THRESHOLD are available from Step 2.

print("\n3. Decision:")

if min_distance <= THRESHOLD:
    # D_min is within the acceptable range.
    final_decision = f"ASSIGNED_TO_GROUP"
    print(f"   Decision: ASSIGN to existing Group {closest_cluster}")
    
elif min_distance > THRESHOLD:
    # D_min is outside the acceptable range (Excessive Deviation).
    final_decision = "EXCESSIVE_DEVIATION_NEW_GROUP"
    print(f"   Decision: FLAG as OUTLIER/NEW GROUP")
    print(f"   ACTION: Trigger periodic batch re-clustering.")

# Example structure for the result
result_dict = {
    "file": "101.txt",
    "D_min": min_distance,
    "Threshold": THRESHOLD,
    "Decision": final_decision,
    "Assigned_Group": closest_cluster
}
# print("\nFinal Result Summary:", result_dict)