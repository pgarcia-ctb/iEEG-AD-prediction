"""
Title: Support Vector Machine (SVM) Model Training and Optimization

Author: Pablo García-Peña  
Affiliation: Centre for Biomedical Technology (CTB), Experimental Neurology Laboratory  
Date: April 08, 2025

Description:  
This script trains a Support Vector Machine (SVM) classifier for genotype prediction (WT vs. 5XFAD) based on feature combinations selected through previous statistical analysis. The workflow includes:  
- Loading and preprocessing genotype data from Excel files  
- Scaling features using StandardScaler  
- Hyperparameter optimization via GridSearchCV with cross-validation to select the best SVM model  
- Evaluation of model performance using metrics such as accuracy, ROC-AUC, PR-AUC, precision, recall, F1-score, and specificity  
- Saving the best trained SVM model and scaler for future predictions  

The script outputs the model’s performance results, including confusion matrix and other metrics, and stores the best SVM model and scaler as serialized files (`best_svm_model.pkl` and `svm_scaler.pkl`). Additionally, it saves the optimal features used for the model in a new Excel file (`new_data_for_prediction.xlsx`).

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
- Trained SVM model (best_svm_model.pkl)  
- Scaler object (svm_scaler.pkl) for feature scaling in future predictions  
- Performance metrics and confusion matrix for the best SVM model  
- Excel file storing optimal features for future predictions  

Usage:  
Ensure data.xlsx and selected_features.xlsx are in the working directory and run the script.  
The output will display model performance and save the best model and scaler for future use.
"""

# Import necessary libraries
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc, average_precision_score, precision_score, recall_score, f1_score
import joblib

# Load the dataset and the optimal features from Excel files
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

# Initialize variables for tracking the best kernel configuration
best_kernel_accuracy = 0
best_kernel_metrics = None
best_kernel = None
best_accuracy_std = 0

# Loop over each kernel type for SVM to find the best performing one
for kernel in ['linear', 'poly', 'rbf']:
    # Define the parameter grid for GridSearchCV depending on the kernel type
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001] if kernel == 'rbf' else ['scale'],
        'kernel': [kernel]
    }

    # Initialize GridSearchCV to search over hyperparameters and perform cross-validation
    grid_search = GridSearchCV(
        SVC(probability=True),
        param_grid,
        scoring='accuracy',
        cv=cv_strategy,
        n_jobs=-1
    )

    # Fit the model using grid search to find the best hyperparameters
    grid_search.fit(X_scaled, y)

    # Retrieve the best model and its associated metrics
    best_svc = grid_search.best_estimator_
    mean_accuracy = grid_search.best_score_
    accuracy_std = grid_search.cv_results_['std_test_score'][grid_search.best_index_]
    best_params = grid_search.best_params_

    # Update the global best model if the current kernel performs better
    if mean_accuracy > best_kernel_accuracy:
        best_kernel_accuracy = mean_accuracy
        best_accuracy_std = accuracy_std
        best_kernel = kernel
        best_kernel_metrics = calculate_model_metrics(best_svc, X_scaled, y)
        best_kernel_metrics.update({
            'Best Params': best_params,
            'Accuracy': mean_accuracy,
            'Accuracy Std Dev': accuracy_std
        })

# Store the results for the selected features
results.append({
    'Kernel': best_kernel,
    'Accuracy': best_kernel_accuracy,
    'Accuracy Std Dev': best_accuracy_std,
    'C': best_kernel_metrics['Best Params']['C'],
    'Gamma': best_kernel_metrics['Best Params'].get('gamma', 'N/A'),
    'ROC-AUC': best_kernel_metrics['ROC-AUC'],
    'PR-AUC': best_kernel_metrics['PR-AUC'],
    'Precision': best_kernel_metrics['Precision'],
    'Recall (Sensitivity)': best_kernel_metrics['Recall (Sensitivity)'],
    'F1-Score (Balance)': best_kernel_metrics['F1-Score (Balance)'],
    'Specificity': best_kernel_metrics['Specificity']
})

# Update the global best model if the current kernel performs better
if (best_kernel_accuracy > highest_accuracy and
    best_accuracy_std < lowest_accuracy_std_dev and
    best_kernel_metrics['Recall (Sensitivity)'] > highest_sensitivity and
    best_kernel_metrics['Specificity'] > highest_specificity):
    
    highest_accuracy = best_kernel_accuracy
    lowest_accuracy_std_dev = best_accuracy_std
    highest_sensitivity = best_kernel_metrics['Recall (Sensitivity)']
    highest_specificity = best_kernel_metrics['Specificity']
    best_model = grid_search.best_estimator_
    best_combination = selected_features
    best_combination_number = best_combination_number
    best_metrics = best_kernel_metrics
    best_scaler = scaler

# Save the best SVM model and the scaler object for future use
# Ensure the features used by the best model are consistent with the scaler
X_best = data[best_combination]

# Fit the scaler only on the best combination of features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_best)

# Save the best SVM model and the scaler object for future use
joblib.dump(best_model, "best_svm_model.pkl")
joblib.dump(scaler, 'svm_scaler.pkl')

# Print the features used by the best model and the scaler
print(f"Features used by the best model (number of features: {len(best_combination)}): {best_combination}")
print(f"Features scaled (columns used): {scaler.feature_names_in_}")

# Display the results for the selected features
print(f"\nResults for the selected features: Kernel = {results[0]['Kernel']}, C = {results[0]['C']}, Gamma = {results[0]['Gamma']}, "
      f"Accuracy = {results[0]['Accuracy']:.4f}, Accuracy Std Dev = {results[0]['Accuracy Std Dev']:.4f}, "
      f"ROC-AUC = {results[0]['ROC-AUC']:.4f}, PR-AUC = {results[0]['PR-AUC']:.4f}, Precision = {results[0]['Precision']:.4f}, "
      f"Recall (Sensitivity) = {results[0]['Recall (Sensitivity)']:.4f}, F1-Score (Balance) = {results[0]['F1-Score (Balance)']:.4f}, "
      f"Specificity = {results[0]['Specificity']:.4f}")

# Display the metrics for the best performing model
print(f"Variables (Number of variables: {len(best_combination)}): {best_combination}")
print(f"Kernel: {best_metrics['Best Params']['kernel']}")
print(f"C: {best_metrics['Best Params']['C']}")
print(f"Gamma: {best_metrics['Best Params'].get('gamma', 'N/A')}")
print(f"Accuracy: {highest_accuracy:.4f}")
print(f"Accuracy Standard Deviation (Stability): {best_metrics['Accuracy Std Dev']:.4f}")
print(f"ROC-AUC: {best_metrics['ROC-AUC']:.4f}")
print(f"PR-AUC: {best_metrics['PR-AUC']:.4f}")
print(f"Precision: {best_metrics['Precision']:.4f}")
print(f"Recall (Sensitivity): {best_metrics['Recall (Sensitivity)']:.4f}")
print(f"F1-Score (Balance): {best_metrics['F1-Score (Balance)']:.4f}")
print(f"Specificity: {best_metrics['Specificity']:.4f}")
print(f"Confusion Matrix:\n{best_metrics['Confusion Matrix']}")
print("\nModeling complete.")
print("The best model has been saved as 'best_svm_model.pkl' and 'svm_scaler.pkl'.")

# Save the optimal combination of variables (column names) for prediction in a new Excel file
optimal_combination_df = pd.DataFrame(columns=best_combination)
optimal_combination_df.to_excel("new_data_for_prediction.xlsx", index=False, startrow=0, startcol=0)
print("An empty Excel file with only the header row of the optimal combination of variables has been saved as 'new_data_for_prediction.xlsx'. "
      "Please complete it with new individuals to be predicted by the model.")