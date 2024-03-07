import numpy as np
confusion_matrices = [[4885,1899,1103,2297],
[7918,238,1939,89],
[4426,2596,1191,1971],
[8074,516,1016,578]]

# Initialize arrays to store metrics for each class
precisions = []
recalls = []
f1_scores = []

# Calculate metrics for each class
for matrix in confusion_matrices:
    # Convert to numpy array if not already
    matrix = np.array(matrix)
    
    TN, FP, FN, TP = matrix.ravel()
    
    # Precision: TP / (TP + FP)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    precisions.append(precision)
    
    # Recall: TP / (TP + FN)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    recalls.append(recall)
    
    # F1 Score: 2 * (precision * recall) / (precision + recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    f1_scores.append(f1)
    print(matrix)
# Calculate averages
avg_precision = np.mean(precisions)
avg_recall = np.mean(recalls)
avg_f1_score = np.mean(f1_scores)

print("Average Precision:", avg_precision)
print("Average Recall:", avg_recall)
print("Average F1 Score:", avg_f1_score)
