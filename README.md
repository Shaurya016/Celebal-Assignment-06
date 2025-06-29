# Celebal-Assignment-06
In this task, we have to train multiple machine learning models and evaluate their performance using metrics such as accuracy, precision, recall, and F1-scoreand implement hyperparameter tuning techniques like GridSearchCV and RandomizedSearchCV to optimize model parameters. Also we have to analyze the results to select the best-performing model.

# 🧠 Breast Cancer Classification using Machine Learning
This repository contains a machine learning project to classify breast cancer tumors as **malignant** or **benign** using various classification models and hyperparameter tuning techniques. The project is based on the **Breast Cancer Wisconsin (Diagnostic)** dataset.

## 📁 Dataset

- **Source**: [UCI Machine Learning Repository](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)))
- **File**: `data.csv`
- **Samples**: 569
- **Features**: 30 numerical features
- **Target**: `diagnosis` (M = malignant, B = benign)

## 🔧 Project Workflow

1. **Data Loading & Cleaning**
   - Removed `id` and `Unnamed: 32` columns
   - Encoded `diagnosis`: `M` → 1, `B` → 0

2. **Data Preprocessing**
   - Split data into training and testing sets
   - Applied feature scaling using `StandardScaler`

3. **Model Training**
   - Trained 5 baseline models:
     - Logistic Regression
     - Support Vector Machine (SVM)
     - Decision Tree
     - Random Forest
     - Naive Bayes

4. **Hyperparameter Tuning**
   - Used `GridSearchCV` for SVM
   - Used `RandomizedSearchCV` for Random Forest

5. **Evaluation Metrics**
   - Accuracy
   - Precision
   - Recall
   - F1 Score

6. **Visualization**
   - Correlation heatmap
   - Class distribution
   - Feature importance
   - Pairplot
   - Boxplots by diagnosis
   - Confusion matrix


