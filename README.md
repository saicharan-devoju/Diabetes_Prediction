# Diabetes Prediction System

This notebook demonstrates a machine learning pipeline to predict the onset of diabetes based on diagnostic measurements. The system uses a Support Vector Machine (SVM) classifier after standardizing the input data.

## Table of Contents
1.  [Project Overview](#project-overview)
2.  [Dataset](#dataset)
3.  [Dependencies](#dependencies)
4.  [Data Collection and Analysis](#data-collection-and-analysis)
5.  [Data Standardization](#data-standardization)
6.  [Train-Test Split](#train-test-split)
7.  [Model Training](#model-training)
8.  [Model Evaluation](#model-evaluation)
9.  [Making a Predictive System](#making-a-predictive-system)

## Project Overview
This project aims to build a predictive model to identify whether a person has diabetes or not, based on several health indicators. The process involves loading a dataset, performing exploratory data analysis, standardizing the features, splitting the data into training and testing sets, training an SVM classifier, and evaluating its performance.

## Dataset
The dataset used is the Pima Indians Diabetes Database, which contains diagnostic measurements for 768 female patients. The 'Outcome' column indicates whether the patient has diabetes (1) or not (0).

**Features:**
*   `Pregnancies`: Number of times pregnant
*   `Glucose`: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
*   `BloodPressure`: Diastolic blood pressure (mm Hg)
*   `SkinThickness`: Triceps skin fold thickness (mm)
*   `Insulin`: 2-Hour serum insulin (mu U/ml)
*   `BMI`: Body mass index (weight in kg/(height in m)^2)
*   `DiabetesPedigreeFunction`: Diabetes pedigree function
*   `Age`: Age (years)

**Target Variable:**
*   `Outcome`: Class variable (0 or 1) indicating non-diabetic or diabetic, respectively.

## Dependencies
The following Python libraries are required:
*   `numpy`
*   `pandas`
*   `sklearn` (for StandardScaler, train_test_split, svm, accuracy_score)
*   `kagglehub` (for downloading the dataset, if applicable)

## Data Collection and Analysis
*   The dataset is loaded using `pandas.read_csv`.
*   Initial exploration includes viewing the first few rows (`.head()`), getting statistical measures (`.describe()`), and checking the distribution of the target variable (`.value_counts()`).
*   The mean values for each feature are grouped by the 'Outcome' to observe differences between diabetic and non-diabetic individuals.

## Data Standardization
*   Features (X) are separated from the target variable (Y).
*   `StandardScaler` from `sklearn.preprocessing` is used to standardize the features. This is crucial for SVM models which are sensitive to the scale of the input features.

## Train-Test Split
*   The standardized data is split into training and testing sets using `train_test_split`.
*   A `test_size` of 20% is used, and `stratify=Y` ensures that the proportion of diabetic/non-diabetic individuals is maintained in both sets. `random_state` is set for reproducibility.

## Model Training
*   A Support Vector Machine (SVM) classifier with a `linear` kernel is initialized.
*   The classifier is trained on the `X_train` and `Y_train` data.

## Model Evaluation
*   **Accuracy Score on Training Data:** The model's performance is evaluated on the training data to check for overfitting.
*   **Accuracy Score on Test Data:** The model's performance is evaluated on unseen test data to get an unbiased estimate of its generalization capability.

## Making a Predictive System
*   A sample `input_data` (new patient's measurements) is provided.
*   This input data is converted to a NumPy array, reshaped, and then standardized using the *same scaler* that was fitted on the training data.
*   The standardized input is fed to the trained `classifier` to get a `prediction`.
*   The system then prints whether the predicted individual is diabetic or not based on the model's output.
