"""
Title: Random Forest (RF) Model Training and Optimization

Author: Pablo García-Peña
Affiliation: Centre for Biomedical Technology (CTB), Experimental Neurology Laboratory
Date: April 08, 2025

Description:
This script trains a Random Forest (RF) classifier for genotype prediction (WT vs. 5XFAD) based on feature combinations selected through previous statistical analysis. The workflow includes:
- Loading and preprocessing genotype data from Excel files
- Scaling features using StandardScaler
- Hyperparameter optimization via GridSearchCV with cross-validation to select the best RF model
- Calculation and display of performance metrics, including accuracy, ROC-AUC, PR-AUC, precision, recall, F1-score, and specificity
- Saving the best trained RF model and scaler for future predictions

The script also outputs the model’s performance results and stores the best RF model and scaler as serialized files (`best_rf_model.pkl` and `rf_scaler.pkl`).

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
- Trained RF model (best_rf_model.pkl)  
- Scaler object (rf_scaler.pkl) for feature scaling in future predictions  
- Performance metrics and confusion matrix for the best RF model  
- Excel file storing optimal features for future predictions  

Usage:  
Ensure data.xlsx and selected_features.xlsx are in the working directory and run the script.  
The output will display model performance and save the best model and scaler for future use.
"""

# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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

# Function to calculate various performance metrics for a trained model
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

# Initialize Random Forest model with OOB scoring enabled
rf_model = RandomForestClassifier(random_state=1, n_jobs=-1, oob_score=True)

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV to search over hyperparameters and perform cross-validation
grid_search = GridSearchCV(
    rf_model,
    param_grid,
    scoring='accuracy',
    cv=cv_strategy,
    n_jobs=-1
)

# Fit the model using grid search to find the best hyperparameters
grid_search.fit(X_scaled, y)

# Retrieve the best model and its associated metrics
best_rf = grid_search.best_estimator_
mean_accuracy = grid_search.best_score_
accuracy_std = grid_search.cv_results_['std_test_score'][grid_search.best_index_]
best_params = grid_search.best_params_

# If the accuracy of the current model is better than the previous best, update the best model info
if mean_accuracy > highest_accuracy:
    highest_accuracy = mean_accuracy
    lowest_accuracy_std_dev = accuracy_std
    best_model = grid_search.best_estimator_
    best_combination = selected_features
    best_combination_number = 1  # Update accordingly
    best_metrics = calculate_model_metrics(best_rf, X_scaled, y)

    # Store the results for the current combination
    results.append({
        'Combination Number': best_combination_number,
        'Accuracy': mean_accuracy,
        'Accuracy Std Dev': accuracy_std,
        'n_estimators': best_params['n_estimators'],
        'max_depth': best_params['max_depth'],
        'min_samples_split': best_params['min_samples_split'],
        'min_samples_leaf': best_params['min_samples_leaf'],
        'ROC-AUC': best_metrics['ROC-AUC'],
        'PR-AUC': best_metrics['PR-AUC'],
        'Precision': best_metrics['Precision'],
        'Recall (Sensitivity)': best_metrics['Recall (Sensitivity)'],
        'F1-Score (Balance)': best_metrics['F1-Score (Balance)'],
        'Specificity': best_metrics['Specificity'],
        'OOB Error': 1 - best_rf.oob_score_
    })

# Save the best Random Forest model and the scaler object for future use
joblib.dump(best_model, "best_rf_model.pkl")
joblib.dump(scaler, 'rf_scaler.pkl')

# Print the features used by the best model and the scaler
print(f"Features used by the best model (number of features: {len(best_combination)}): {best_combination}")
print(f"Features scaled (columns used): {scaler.feature_names_in_}")

# Display the results for all tested combinations
print("\nResults for each combination tested:")
for result in results:
    print(f"Combination {result['Combination Number']}: n_estimators = {result['n_estimators']}, max_depth = {result['max_depth']}, "
          f"min_samples_split = {result['min_samples_split']}, min_samples_leaf = {result['min_samples_leaf']}, "
          f"Accuracy = {result['Accuracy']:.4f}, Accuracy Std Dev = {result['Accuracy Std Dev']:.4f}, "
          f"ROC-AUC = {result['ROC-AUC']:.4f}, PR-AUC = {result['PR-AUC']:.4f}, Precision = {result['Precision']:.4f}, "
          f"Recall (Sensitivity) = {result['Recall (Sensitivity)']:.4f}, F1-Score (Balance) = {result['F1-Score (Balance)']:.4f}, "
          f"Specificity = {result['Specificity']:.4f}, OOB Error = {result['OOB Error']:.4f}")

# Display the metrics for the best performing model
print(f"Variables (Number of variables: {len(best_combination)}): {best_combination}")
print(f"Best Parameters: {best_params}")
print(f"Accuracy: {highest_accuracy:.4f}")
print(f"Accuracy Standard Deviation (Stability): {accuracy_std:.4f}")
print(f"ROC-AUC: {best_metrics['ROC-AUC']:.4f}")
print(f"PR-AUC: {best_metrics['PR-AUC']:.4f}")
print(f"Precision: {best_metrics['Precision']:.4f}")
print(f"Recall (Sensitivity): {best_metrics['Recall (Sensitivity)']:.4f}")
print(f"F1-Score (Balance): {best_metrics['F1-Score (Balance)']:.4f}")
print(f"Specificity: {best_metrics['Specificity']:.4f}")
print(f"Confusion Matrix:\n{best_metrics['Confusion Matrix']}")
print(f"OOB Error: {1 - best_rf.oob_score_:.4f}")

# Save the optimal combination of variables (column names) for prediction in a new Excel file
# optimal_combination_df = pd.DataFrame(columns=best_combination)
# optimal_combination_df.to_excel("new_data_for_prediction.xlsx", index=False, startrow=0, startcol=0)
# print("An empty Excel file with only the header row of the optimal combination of variables has been saved as 'new_data_for_prediction.xlsx'. "
#       "Please complete it with new individuals to be predicted by the model.")