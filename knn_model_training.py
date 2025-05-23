"""
Title: K-Nearest Neighbors (KNN) Model Training and Optimization

Author: Pablo García-Peña
Affiliation: Centre for Biomedical Technology (CTB), Experimental Neurology Laboratory  
Date: April 08, 2025

Description:  
This script trains a K-Nearest Neighbors (KNN) classifier for genotype prediction (WT vs. 5XFAD) based on feature combinations selected through previous statistical analysis. The workflow includes:  
- Loading and preprocessing genotype data from Excel files  
- Scaling features using StandardScaler  
- Hyperparameter optimization via GridSearchCV with cross-validation to select the best KNN model  
- Calculation and display of performance metrics, including accuracy, ROC-AUC, PR-AUC, precision, recall, F1-score, and specificity  
- Saving the best trained KNN model and scaler for future predictions  

The script also outputs the model’s performance results and stores the best KNN model and scaler as serialized files (`best_knn_model.pkl` and `knn_scaler.pkl`).

Article Reference:  
This code is part of the analysis presented in the article  
"Slow-Wave–Modulated High-Frequency Oscillations Reveal Early Network Abnormalities in Asymptomatic 5XFAD Mice"  
by Pablo García-Peña, Milagros Ramos, Juan M. López, Ricardo Martinez-Murillo, Guillermo de Arcas and Daniel González-Nieto.  

License:  
This work is licensed under the Apache License 2.0.  
For full details, see the LICENSE file in this repository or visit: https://www.apache.org/licenses/LICENSE-2.0.

Dependencies:
Python 3.9+
Required libraries:
- pandas 2.2.3
- scikit-learn 1.6.0
- joblib 1.4.2

Input:  
- Excel file (data.xlsx) containing numerical variables and a categorical column Genotype (WT = 0, 5XFAD = 1)  
- Excel file (selected_features.xlsx) with selected features for the model  

Output:  
- Trained KNN model (best_knn_model.pkl)  
- Scaler object (knn_scaler.pkl) for feature scaling in future predictions  
- Performance metrics and confusion matrix for the best KNN model   
- Excel file storing optimal features for future predictions  

Usage:  
Ensure data.xlsx and selected_features.xlsx are in the working directory and run the script.  
The output will display model performance and save the best model and scaler for future use.
"""

# Import necessary libraries
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc, average_precision_score, precision_score, recall_score, f1_score
import joblib

# Load the dataset and optimal combinations from Excel files
data = pd.read_excel("data.xlsx")
data['Genotype'] = data['Genotype'].map({'WT': 0, '5XFAD': 1})
combinations_df = pd.read_excel("selected_features.xlsx")

# Extract the selected features from the 'Selected Features' column
selected_features = combinations_df['Selected Features'].dropna().tolist()

# Define cross-validation strategy: 5 splits, 3 repeats, with random state for reproducibility
cv_strategy = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

# Initialize variables to store the best model and related information
best_model = None
best_combination = None
best_combination_number = None
best_metrics = None
highest_accuracy = 0
lowest_accuracy_std_dev = float('inf')
highest_sensitivity = 0
highest_specificity = 0
results = []

# Extract the selected features and the target variable
X = data[selected_features]
y = data['Genotype']
scaler = StandardScaler()

# Scale the data
X_scaled = scaler.fit_transform(X)

# Function to calculate metrics for the given model
def calculate_model_metrics(model, X_scaled, y):
    """Calculate performance metrics for a trained model."""
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]
    cm = confusion_matrix(y, y_pred)
    tp, fn, fp, tn = cm.ravel()

    metrics = {
        'Confusion Matrix': cm,
        'ROC-AUC': auc(*roc_curve(y, y_prob)[:2]),
        'PR-AUC': average_precision_score(y, y_prob),
        'Precision': precision_score(y, y_pred),
        'Recall (Sensitivity)': recall_score(y, y_pred),
        'F1-Score (Balance)': f1_score(y, y_pred),
        'Accuracy': (tp + tn) / (tp + tn + fp + fn),
        'Specificity': tn / (tn + fp)
    }
    return metrics

# Initialize variables for tracking the best Gradient Boosting model configuration for this combination
best_model_accuracy = 0
best_model_metrics = None
best_model_params = None
best_accuracy_std = 0

# Define hyperparameter grid for GridSearchCV
param_grid = {
    'n_neighbors': [1, 3, 5],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

# Initialize GridSearchCV with KNN classifier
grid_search = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    scoring='accuracy',
    cv=cv_strategy,
    n_jobs=-1
)

# Fit the model using grid search
grid_search.fit(X_scaled, y)

# Retrieve the best model and its associated metrics
best_knn = grid_search.best_estimator_
mean_accuracy = grid_search.best_score_
accuracy_std = grid_search.cv_results_['std_test_score'][grid_search.best_index_]
best_params = grid_search.best_params_

# If the accuracy of the current Gradient Boosting model is better than the previous best, update the best model info
if mean_accuracy > best_model_accuracy:
    best_model_accuracy = mean_accuracy
    best_accuracy_std = accuracy_std
    best_model_params = best_params
    best_model_metrics = calculate_model_metrics(best_knn, X_scaled, y)

# Store the results for the current combination
results.append({
    'Combination Number': best_combination_number,
    'Accuracy': best_model_accuracy,
    'Accuracy Std Dev': best_accuracy_std,
    'n_neighbors': best_params['n_neighbors'],
    'weights': best_params['weights'],
    'metric': best_params['metric'],
    'ROC-AUC': best_model_metrics['ROC-AUC'],
    'PR-AUC': best_model_metrics['PR-AUC'],
    'Precision': best_model_metrics['Precision'],
    'Recall (Sensitivity)': best_model_metrics['Recall (Sensitivity)'],
    'F1-Score (Balance)': best_model_metrics['F1-Score (Balance)'],
    'Specificity': best_model_metrics['Specificity']
})

# Update the global best model if the current Gradient Boosting model performs better
if (best_model_accuracy > highest_accuracy and
    best_accuracy_std < lowest_accuracy_std_dev and
    best_model_metrics['Recall (Sensitivity)'] > highest_sensitivity and
    best_model_metrics['Specificity'] > highest_specificity):
        
    highest_accuracy = best_model_accuracy
    lowest_accuracy_std_dev = best_accuracy_std
    highest_sensitivity = best_model_metrics['Recall (Sensitivity)']
    highest_specificity = best_model_metrics['Specificity']
    best_model = grid_search.best_estimator_
    best_combination = selected_features
    best_combination_number = best_combination_number
    best_metrics = best_model_metrics
    best_scaler = scaler

# Save the best KNN model and the scaler object for future use
# Ensure the features used by the best model are consistent with the scaler
X_best = data[best_combination]

# Fit the scaler only on the best combination of features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_best)

# Save the best KNN model to a file
joblib.dump(best_model, "best_knn_model.pkl")
joblib.dump(scaler, 'knn_scaler.pkl')

# Print the features used by the best model and the scaler
print(f"Features used by the best model (number of features: {len(best_combination)}): {best_combination}")
print(f"Features scaled (columns used): {scaler.feature_names_in_}")

# Display results for each combination tested
print("\nResults for each combination tested:")
for result in results:
    print(f"Combination {results[0]['Combination Number']}: n_neighbors = {results[0]['n_neighbors']}, weights = {results[0]['weights']}, metric = {results[0]['metric']}, "
          f"Accuracy = {results[0]['Accuracy']:.4f}, Accuracy Std Dev = {results[0]['Accuracy Std Dev']:.4f}, "
          f"ROC-AUC = {results[0]['ROC-AUC']:.4f}, PR-AUC = {results[0]['PR-AUC']:.4f}, Precision = {results[0]['Precision']:.4f}, "
          f"Recall (Sensitivity) = {results[0]['Recall (Sensitivity)']:.4f}, F1-Score (Balance) = {results[0]['F1-Score (Balance)']:.4f}, "
          f"Specificity = {results[0]['Specificity']:.4f}")

# Display the metrics for the best performing model
print(f"Variables (Number of variables: {len(best_combination)}): {best_combination}")
print(f"n_neighbors: {best_model_params['n_neighbors']}")
print(f"weights: {best_model_params['weights']}")
print(f"metric: {best_model_params['metric']}")
print(f"Accuracy: {highest_accuracy:.4f}")
print(f"Accuracy Standard Deviation (Stability): {best_accuracy_std:.4f}")
print(f"ROC-AUC: {best_metrics['ROC-AUC']:.4f}")
print(f"PR-AUC: {best_metrics['PR-AUC']:.4f}")
print(f"Precision: {best_metrics['Precision']:.4f}")
print(f"Recall (Sensitivity): {best_metrics['Recall (Sensitivity)']:.4f}")
print(f"F1-Score (Balance): {best_metrics['F1-Score (Balance)']:.4f}")
print(f"Specificity: {best_metrics['Specificity']:.4f}")
print(f"Confusion Matrix:\n{best_metrics['Confusion Matrix']}")
print("\nModeling complete.")

# Save the optimal combination of variables (column names) for prediction in a new Excel file
# optimal_combination_df = pd.DataFrame(columns=best_combination)
# optimal_combination_df.to_excel("new_data_for_prediction.xlsx", index=False, startrow=0, startcol=0)
# print("An empty Excel file with only the header row of the optimal combination of variables has been saved as 'new_data_for_prediction.xlsx'. "
#       "Please complete it with new individuals to be predicted by the model.")