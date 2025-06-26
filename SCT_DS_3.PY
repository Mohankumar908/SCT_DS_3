import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Load Dataset
df = pd.read_csv('bank.csv', sep=';')  # Ensure bank.csv is in your working directory

# Display basic info
print(df.head())
print(df['y'].value_counts())

# Encode target variable
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# One-hot encode categorical columns
df_encoded = pd.get_dummies(df.drop('y', axis=1))

# Final features and labels
X = df_encoded.values
y = df['y'].values

# Train-test split (80/20)
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Gini Impurity
def gini_index(y):
    counts = Counter(y)
    impurity = 1
    for label in counts:
        prob = counts[label] / len(y)
        impurity -= prob ** 2
    return impurity

# Best Split Function
def best_split(X, y):
    best_feature = None
    best_threshold = None
    best_gain = 0
    current_gini = gini_index(y)

    for feature in range(X.shape[1]):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            left_mask = X[:, feature] <= threshold
            right_mask = X[:, feature] > threshold

            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue

            left_gini = gini_index(y[left_mask])
            right_gini = gini_index(y[right_mask])

            p = float(np.sum(left_mask)) / len(y)
            gain = current_gini - (p * left_gini + (1 - p) * right_gini)

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold

# Decision Stump (One Split Tree)
def predict_stump(X, feature, threshold):
    return np.where(X[:, feature] <= threshold, 0, 1)

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

# Train
feature, threshold = best_split(X_train, y_train)
print(f"Best Feature Index: {feature}, Threshold: {threshold}")

# Predict
y_pred = predict_stump(X_test, feature, threshold)

# Accuracy
acc = accuracy(y_test, y_pred)
print(f"Test Accuracy: {acc * 100:.2f}%")

# Confusion Matrix (Manual)
def compute_confusion_matrix(y_true, y_pred):
    TP = TN = FP = FN = 0
    for actual, predicted in zip(y_true, y_pred):
        if actual == 1 and predicted == 1:
            TP += 1
        elif actual == 0 and predicted == 0:
            TN += 1
        elif actual == 0 and predicted == 1:
            FP += 1
        elif actual == 1 and predicted == 0:
            FN += 1
    return np.array([[TN, FP], [FN, TP]])

cm = compute_confusion_matrix(y_test, y_pred)

# Plot Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Basic Classification Report
TN, FP, FN, TP = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
precision = TP / (TP + FP) if TP + FP else 0
recall = TP / (TP + FN) if TP + FN else 0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

print("\nManual Classification Report:")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1 Score:  {f1:.2f}")
