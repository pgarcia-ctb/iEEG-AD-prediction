"""
Title: Recursive Feature Elimination with Cross-Validation (RFECV) for Feature Selection

Author: Pablo García-Peña
Affiliation: Centre for Biomedical Technology (CTB), Experimental Neurology Laboratory
Date: April 08, 2025

Description:
This script performs feature selection for genotype data extracted from an Excel file.
It applies statistical tests followed by Recursive Feature Elimination with Cross-Validation (RFECV)
using a RandomForestClassifier to identify the optimal subset of features that best predict genotype.

The selected features are saved in an Excel file (`selected_features.xlsx`) for further analysis.

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
- numpy 2.1.3
- scikit-learn 1.6.0
- scipy 1.9.0

Input:  
- Excel file (data.xlsx) containing numerical variables and a categorical column Genotype (WT = 0, 5XFAD = 1)

Output:
- Optimized feature subset selected via RFECV
- Excel file (selected_features.xlsx) storing selected features

Usage:  
Ensure data.xlsx is in the working directory and run the script.
The output will display selected features and save the optimal feature set to an Excel file.
"""

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from scipy import stats

# Load dataset from an Excel file
data = pd.read_excel("data.xlsx")

# Map the Genotype variable (0 = WT, 1 = 5XFAD)
data['Genotype'] = data['Genotype'].map({'WT': 0, '5XFAD': 1})

# Extract numeric variables excluding Genotype
X = data.select_dtypes(include=[np.number]).drop(columns='Genotype')
y = data['Genotype']

# Statistical analysis functions
def test_normality(group):
    """Shapiro-Wilk normality test"""
    return stats.shapiro(group).pvalue

def test_variance(group1, group2):
    """Levene’s test for equal variances"""
    return stats.levene(group1, group2).pvalue

def select_test(data, variable):
    """Determines and applies the appropriate statistical test"""
    group1 = data[data['Genotype'] == 0][variable]
    group2 = data[data['Genotype'] == 1][variable]

    p_norm1, p_norm2 = test_normality(group1), test_normality(group2)
    p_var = test_variance(group1, group2)

    if p_norm1 > 0.05 and p_norm2 > 0.05 and p_var > 0.05:
        _, p_value = stats.ttest_ind(group1, group2)
        test_used = "t-test"
    else:
        _, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        test_used = "Mann-Whitney U"

    return {"Variable": variable, "p-value": p_value, "Test Used": test_used}

# Apply statistical tests
variables = X.columns  # List of variables to test
results_df = pd.DataFrame([select_test(data, var) for var in variables])

# Filter significant variables (p-value < 0.05)
significant_df = results_df[results_df['p-value'] < 0.05].sort_values(by='p-value')

# Print significant variables
print("\nSignificant variables:")
print(significant_df.to_string(index=False))

# Feature Selection using RFECV with Random Forest
selected_variables = significant_df['Variable'].to_list()  # Select significant variables
X_selected = X[selected_variables]  # Subset data to significant features

model = RandomForestClassifier(n_estimators=100, random_state=42)
rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(5), scoring='accuracy')
rfecv.fit(X_selected, y)

# Selected features using RFECV
selected_features = X_selected.columns[rfecv.support_]
print("\nSelected Features using RFECV:")
print(selected_features.to_list())

# Save selected features
pd.DataFrame(selected_features, columns=["Selected Features"]).to_excel("selected_features.xlsx", index=False)
print("\nSelected Features have been saved in 'selected_features.xlsx'.")