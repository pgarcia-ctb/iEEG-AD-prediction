"""
Title: Feature Selection Using p-value Threshold and Akaike Information Criterion (AIC)

Author: Pablo García-Peña  
Affiliation: Centre for Biomedical Technology (CTB), Experimental Neurology Laboratory  
Date: April 08, 2025

Description:  
This script performs feature selection in two stages:  
1. Filters variables by statistical significance (p < 0.05) using t-test or Mann-Whitney U.  
2. Applies AIC to all combinations of significant variables to select the optimal subset using OLS regression.  

The selected features are saved in an Excel file (selected_features.xlsx) for further analysis.  

Article Reference:  
This code is part of the analysis presented in the article  
"Slow-Wave–Modulated High-Frequency Oscillations Reveal Early Network Abnormalities in Asymptomatic 5XFAD Mice"  
by Pablo García-Peña, Milagros Ramos, Juan M. López, Ricardo Martinez-Murillo, Guillermo de Arcas and Daniel González-Nieto.  

License:  
This work is licensed under the Apache License 2.0.  
For full details, see the LICENSE file in this repository or visit: https://www.apache.org/licenses/LICENSE-2.0.

Dependencies:  
- Python 3.9+  
- pandas 2.2.3  
- numpy 2.1.3  
- statsmodels 0.14.4  
- scipy 1.13.0  

Input:  
- Excel file (data.xlsx) containing numerical variables and a categorical column Genotype (WT = 0, 5XFAD = 1)  

Output:  
- Optimized feature subset selected via AIC  
- Excel file (selected_features.xlsx) storing selected features  
"""

# Import necessary libraries
import pandas as pd
import numpy as np
from itertools import combinations
import statsmodels.api as sm
from scipy import stats

# Load dataset
data = pd.read_excel("data.xlsx")
data['Genotype'] = data['Genotype'].map({'WT': 0, '5XFAD': 1})

# Extract numeric variables excluding Genotype
variables = data.select_dtypes(include=[np.number]).columns.drop('Genotype')

# Statistical testing functions
def test_normality(group):
    return stats.shapiro(group).pvalue

def test_variance(group1, group2):
    return stats.levene(group1, group2).pvalue

def select_test(data, variable):
    group1 = data[data['Genotype'] == 0][variable]
    group2 = data[data['Genotype'] == 1][variable]

    p_norm1, p_norm2 = test_normality(group1), test_normality(group2)
    p_var = test_variance(group1, group2)

    if p_norm1 > 0.05 and p_norm2 > 0.05 and p_var > 0.05:
        _, p_value = stats.ttest_ind(group1, group2)
    else:
        _, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')

    return {"Variable": variable, "p-value": p_value}

# Step 1: Apply p-value filtering
results = [select_test(data, var) for var in variables]
results_df = pd.DataFrame(results)

# Filter significant variables (p < 0.05)
significant_vars = results_df[results_df['p-value'] < 0.05]['Variable'].tolist()

if not significant_vars:
    print("No variables passed the p-value threshold (p < 0.05).")
else:
    print("\nSignificant variables (p < 0.05):")
    print(results_df[results_df['Variable'].isin(significant_vars)].sort_values('p-value').to_string(index=False))

    # Step 2: AIC-based selection on significant variables
    X = data[significant_vars]
    y = data['Genotype']
    best_aic = float('inf')
    selected_features = None

    for k in range(1, len(significant_vars) + 1):
        for subset in combinations(significant_vars, k):
            X_subset = sm.add_constant(X[list(subset)])
            model = sm.OLS(y, X_subset).fit()
            if model.aic < best_aic:
                best_aic = model.aic
                selected_features = subset

    # Save selected features
    if selected_features:
        pd.DataFrame(selected_features, columns=["Selected Features"]).to_excel("selected_features.xlsx", index=False)
        print("\nSelected Features after AIC optimization:")
        print(selected_features)
        print("\nSelected Features have been saved in 'selected_features.xlsx'.")
    else:
        print("No optimal subset found via AIC among significant variables.")
