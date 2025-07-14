# 🌳 Bank Marketing – Decision Tree Classifier (From Scratch)

This project is part of **Task 03** of the SkillCraft Technology Internship. The goal is to build a **Decision Tree Classifier from scratch** using the Bank Marketing dataset to predict whether a customer will subscribe to a term deposit.

---

## 📁 Dataset

- 📂 **File**: `bank.csv`  
- 📊 **Source**: UCI Machine Learning Repository  
- 🔗 [Bank Marketing Dataset Link](https://archive.ics.uci.edu/ml/datasets/bank+marketing)

---

## 🛠️ Tools Used

- Python (no sklearn for training logic)
- pandas, numpy
- seaborn & matplotlib (for visualization)

---

## 🔍 Project Highlights

### ✅ Preprocessing

- Loaded data and performed label encoding for the target variable `y` (`yes` → 1, `no` → 0).
- Applied **one-hot encoding** to categorical features.
- Performed a manual **80/20 train-test split**.

---

### 🌳 Decision Tree Logic

- **Gini Index** calculated for impurity.
- **Best Split Selection** done by checking all thresholds across all features.
- Implemented a **decision stump** (one-split tree).
- Used a basic `predict_stump()` function to simulate tree-based classification.

---

### 📈 Evaluation Metrics

- Manually calculated:
  - Confusion Matrix
  - Accuracy
  - Precision
  - Recall
  - F1 Score
- Plotted the confusion matrix using `seaborn`.

---

## 🔍 Sample Output

| Metric     | Value      |
|------------|------------|
| Accuracy   | 88.XX %    |
| Precision  | 0.XX       |
| Recall     | 0.XX       |
| F1 Score   | 0.XX       |

(*Actual values depend on dataset and split*)

---

## 📂 Project Structure

