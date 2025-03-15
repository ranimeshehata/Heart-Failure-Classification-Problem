# Heart Failure Prediction Project

This project focuses on predicting heart failure using various machine learning algorithms. The dataset used is the **Heart Failure Prediction Dataset** from Kaggle, which contains medical information about patients and whether they experienced heart failure.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Algorithms Implemented](#algorithms-implemented)
6. [Results](#results)

---

## Introduction
Heart failure is a critical medical condition that requires early detection for effective treatment. This project aims to build and evaluate machine learning models to predict heart failure based on patient data. The project includes the implementation of several algorithms, including:
- Decision Trees
- Bagging Ensembles
- AdaBoost
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Feedforward Neural Network (FNN)

---

## Dataset
The dataset used in this project is the **Heart Failure Prediction Dataset** from Kaggle. It contains 918 rows and 12 columns, including features such as age, sex, chest pain type, resting blood pressure, cholesterol levels, fasting blood sugar, resting ECG, maximum heart rate, exercise-induced angina, oldpeak, ST slope, and the target variable `HeartDisease`.

### Dataset Features:
- **Age**: Age of the patient
- **Sex**: Gender of the patient (M/F)
- **ChestPainType**: Type of chest pain (ATA, NAP, ASY, TA)
- **RestingBP**: Resting blood pressure
- **Cholesterol**: Serum cholesterol level
- **FastingBS**: Fasting blood sugar (1 if > 120 mg/dl, 0 otherwise)
- **RestingECG**: Resting electrocardiogram results (Normal, ST, LVH)
- **MaxHR**: Maximum heart rate achieved
- **ExerciseAngina**: Exercise-induced angina (Y/N)
- **Oldpeak**: ST depression induced by exercise relative to rest
- **ST_Slope**: Slope of the peak exercise ST segment (Up, Flat, Down)
- **HeartDisease**: Target variable (1: heart disease, 0: normal)

---

## Required Libraries

- kagglehub
- pandas
- numpy
- matplotlib
- scikit-learn
- torch

## Installation
1. Clone the repository:
    ```sh
    git clone <https://github.com/ranimeshehata/Heart-Failure-Classification-Problem>
    ```
2. Install the required Python libraries:
  You can do this using `pip`:

```sh
pip install numpy pandas matplotlib scikit-learn kagglehub torch
```
---

## Usage
- Download the Dataset: The dataset is downloaded using the kagglehub library. 
- Preprocessing: The dataset is preprocessed by normalizing numeric features and encoding categorical features using one-hot encoding.
- Model Training: The project includes the implementation of several machine learning models:
- Evaluation: Each model is evaluated using accuracy, F1-score, and confusion matrices on the training, validation, and test sets.

---

## Algorithms Implemented
1. Decision Tree
A custom Decision Tree classifier is implemented from scratch.

The model is trained with different hyperparameters (max depth, min samples split) and evaluated using accuracy and F1-score.

2. Bagging Ensemble
A Bagging Ensemble model is implemented using bootstrap sampling and majority voting.

The model is evaluated with different numbers of bootstrap samples.

3. AdaBoost Ensemble
An AdaBoost model is implemented using decision stumps as weak learners.

The model is trained with early stopping to prevent overfitting.

4. K-Nearest Neighbors (KNN)
The KNN model is implemented using scikit-learn.

Hyperparameter tuning is performed to find the best k and distance metric.

5. Logistic Regression
Logistic Regression is implemented using scikit-learn.

The regularization parameter C is tuned using the validation set.

6. Feedforward Neural Network (FNN)
A simple feedforward neural network is implemented using PyTorch.

The model is trained with different hidden layer sizes and learning rates.

----

## Results
1. Decision Tree
- Best Hyperparameters: Depth = 5, Min Samples = 2

- Validation Accuracy: 84.78%

- Test Accuracy: 85.33%

2. Bagging Ensemble
- Best Number of Datasets: 10

- Validation Accuracy: 89.13%

- Test Accuracy: 82.61%

3. AdaBoost
- Validation Accuracy: 86.96%

- Test Accuracy: 89.13%

4. K-Nearest Neighbors (KNN)
- Best Hyperparameters: k = 7, Metric = Euclidean

- Validation Accuracy: 89.13%

- Test Accuracy: 88.04%

5. Logistic Regression
- Best Hyperparameters: C = 0.1

- Validation Accuracy: 90.22%

- Test Accuracy: 88.59%

6. Feedforward Neural Network (FNN)
- Best Hyperparameters: Hidden Layer Size = 16, Learning Rate = 0.1

- Validation Accuracy: 88.04%

- Test Accuracy: 89.13%

The results of the different classifiers are saved as images in the `output_confusion_matrices/` directory. These images include:

- Training results
- Validation results
- Test results
