import pandas as pd
import numpy as np

# Making function to find euclidean distance 
def euclidean_distance(instance1, instance2):
    return np.sqrt(np.sum((instance1 - instance2)**2))

# Making function to implement KNN algorithm
def knn(train_data, test_instance, k):
    distances = []
    
    # Calculate distance between test_instance and each instance in train_data
    for i, train_instance in enumerate(train_data):
        dist = euclidean_distance(test_instance, train_instance[:-1])  # excluding class label
        distances.append((i, dist))
    
    # Sort distances list based on distance
    distances.sort(key=lambda x: x[1])
    
    # FEtching indices of k nearest neighbors
    neighbors_indices = [index for index, _ in distances[:k]]
    
    # Get the class labels of k nearest neighbors
    neighbor_labels = [train_data[i][-1] for i in neighbors_indices]
    
    # Return the most common class label among the k nearest neighbors
    return max(set(neighbor_labels), key=neighbor_labels.count)

# Load data
customer_segmentation = pd.read_csv("D:\customer_segmentation.csv")  


# Splitting dataset into training and testing sets from 863 to 1725
# as trainig=850 and test=12 total 862 
starting_value_training=862
ending_value_training=1713
starting_value_test=1713
ending_value_test=1725
train_data = customer_segmentation.iloc[starting_value_training:ending_value_training]  
test_data = customer_segmentation.iloc[starting_value_test:ending_value_test]     

# testing values of k
k_values = [1, 3, 5, 7]

# Iterating in testing
for instance_index, (_, test_instance) in enumerate(test_data.iterrows()):
    print(f"Instance Index: {instance_index + 1714}")
    print(f"Class Label: {test_instance.values[-1]}")
    for k in k_values:
        predicted_label = knn(train_data.values, test_instance.values[:-1], k)
        print(f"K={k}: {predicted_label}")
    print()  
