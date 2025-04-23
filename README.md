# ML-EEG-AD-prediction: Genotype Prediction of Mice from iEEG Signals Using Machine Learning

## Overview
EEG-AD-Predictor is a pioneering machine learning-based tool designed to predict the genotype of mice (WT vs. 5XFAD) from non-invasive intracranial EEG (iEEG) signals. Our study is among the first to evaluate a range of machine learning models trained to predict Alzheimer’s disease (AD) pathology from presymptomatic 5XFAD young mice. By analyzing non-invasive and cost-effective EEG biomarkers, we aim to provide a reliable method for early-stage AD detection. This tool integrates multiple EEG features, including spectral characteristics, high-frequency oscillation (HFO) bursts, and phase-amplitude coupling, to enhance the accuracy of preclinical AD diagnosis.

## Project Objectives
This project demonstrates the potential of using machine learning to analyze EEG data and predict the genotype of mice, distinguishing between wild-type (WT) and 5XFAD mice, which serve as a model for AD. The study develops a robust machine learning pipeline to identify and validate EEG biomarkers, thus improving our understanding of AD pathology in presymptomatic stages. Additionally, our work shows the ability of these models to predict the genotype of animals with blinded classification, reinforcing their potential for AD diagnosis. By incorporating diverse EEG features, the project contributes to the development of non-invasive, cost-effective tools for early detection of AD and other neurodegenerative diseases.

## Setup Instructions

### 1. Prepare the `data.xlsx` file: 
- **Data Structure**: Each row represents a sample. The columns should include extracted features (e.g., power in different frequency bands) and a `Genotype` column.
- **Genotype Column**: Label the samples as either **WT** (wild-type) or **5XFAD** (Alzheimer’s model).

### 2. Feature Selection
Choose from two feature selection methods:
- **AIC Feature Selection:** To select the optimal subset of features using AIC, run the following script: `python feature_analysis_aic.py`.
  This will select features based on the Akaike Information Criterion (AIC) and save the results in `selected_features.xlsx`.
- **RFECV Feature Selection:**: Alternatively, use Recursive Feature Elimination with Cross-Validation (RFECV) by running: `python feature_analysis_rfecv.py`.
  The optimal features will be saved in `selected_features.xlsx`.

### 3. Model Training
After selecting the features, you can train different machine learning models (KNN, SVM, RF, XGB) using the following scripts:
- **KNN Model Training:** Run: `python knn_model_training.py`.
This will train a KNN model and save the best model as `est_knn_model.pkl` and the scaler as `knn_scaler.pkl`.
- **SVM Model Training:** Run: `python svm_model_training.py`.
This will train an SVM model and save the best model as `best_svm_model.pkl` and the scaler as `svm_scaler.pkl`.
- **Random Forest Model Training:** Run: `python rf_model_training.py`.
This will train a Random Forest model and save the best model as `best_rf_model.pkl` and the scaler as `rf_scaler.pkl`.
- **XGBoost Model Training:** Run: `python xgb_model_training.py`.
This will train an XGBoost model and save the best model as `best_xgb_model.pkl` and the scaler as `xgb_scaler.pkl`.

### 4. Prepare New Data for Prediction
Fill the file `new_data_for_prediction.xlsx` with new data that you want to predict. This file should have the same features as the training data but exclude the Genotype column.

### 5. Prediction
Once your new data is prepared, you can make predictions using the trained models. Run the following scripts depending on the model you trained:
- **KNN Prediction:** Run: `python knn_predictor.py`.
- **SVM Prediction:** Run: `python svm_predictor.py`.
- **Random Forest Prediction:** Run: `python rf_predictor.py`.
- **XGBoost Prediction:** Run: `python xgb_predictor.py`.
All of them will save the results in `predictions.xlsx`.

## Support and Contact
For any questions or assistance with the code, please contact: **Pablo García-Peña** (p.garcia@ctb.upm.es).

## Article Reference
This code is part of the analysis presented in the article:
"Slow-Wave–Modulated High-Frequency Oscillations Reveal Early Network Abnormalities in Asymptomatic 5XFAD Mice"
by Pablo García-Peña, Milagros Ramos, Juan M. López, Ricardo Martinez-Murillo, Guillermo de Arcas, and Daniel González-Nieto.

## License
This project is licensed under the Apache License 2.0. See the LICENSE file included in this repository for full details.
