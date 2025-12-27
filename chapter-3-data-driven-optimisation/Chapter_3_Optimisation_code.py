"""
Chapter 3: Concentrate Optimization using Scipy Differential Evolution

This script implements Scipy's differential evolution algorithm for optimizing 
dairy concentrate allocation using a pre-trained Random Forest model.

Method: Differential Evolution with Linear Constraints
- Evolutionary algorithm for global optimization
- Linear constraints to maintain daily budget
- Dynamic bounds based on previous day's solution

Data Requirements:
- full_data.csv - Complete dataset
- train_data.csv, test_data.csv - Train/test splits
- X_train.csv, y_train.csv, X_test.csv, y_test.csv - Feature/target splits
- optimization_data.csv - Daily data for 81 cows over 91 days
- rf_model.pkl - Pre-trained Random Forest model

Repository:
https://github.com/Stansfash/thesis-python_codes/new/main/chapter-3-data-driven-optimisation
"""

#!/usr/bin/env python
# coding: utf-8

# ## Start

# In[4]:


import csv
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


Full_dataframe_130_cows = pd.read_csv("data/file.csv")
df_train_130 = pd.read_csv("data/file.csv")
df_test_130 = pd.read_csv("data/file.csv")
X_train = pd.read_csv("data/file.csv")
X_test = pd.read_csv("data/file.csv")
y_train = pd.read_csv("data/file.csv")
y_test = pd.read_csv("data/file.csv")


# In[2]:


import pickle
from pickle import load
# load the model
loaded_model = load(open('data/file.csv', 'rb'))


# In[7]:


df_81cows_91days_optimisation = pd.read_csv("data/file.csv").drop("Unnamed: 0", axis=1)
df_81cows_91days_optimisation['smp_date'] = df_81cows_91days_optimisation['smp_date'].astype('datetime64[ns]')
print(df_81cows_91days_optimisation.columns)
#Description of the data subset
subset_data_summary = df_81cows_91days_optimisation[["dim","lw", "conc", "my", "Response"]].describe(include="all")
print("Statistical Summary of Cows in the Subset Dataset \n")
print(subset_data_summary)


# In[9]:


### All Variables and Target Variables
new_arrangement = ["smp_date", "id", "lact", "dim", "lw", "conc", "my"]
df_81cows_91days = df_81cows_91days_optimisation[new_arrangement]
df_81cows_91days = pd.DataFrame(df_81cows_91days)
print(df_81cows_91days.shape)
### Response per animal per day
new_arrangement = ["smp_date", "id","Response"]
df_milk_yield_per_concentrate_consumed = df_81cows_91days_optimisation[new_arrangement]
df_milk_yield_per_concentrate_consumed = pd.DataFrame(df_milk_yield_per_concentrate_consumed)
df_milk_yield_per_concentrate_consumed = df_milk_yield_per_concentrate_consumed.set_index('smp_date')
response = df_milk_yield_per_concentrate_consumed.pivot(columns='id')
# By specifying col[1] in below list comprehension
# You can select the stock names under multi-level column
response.columns = [col[1] for col in response.columns]
print(response.shape)
# print(response.columns)


#  ### Looped Optimisation Process

# In[ ]:


import pandas as pd
import numpy as np
import time
import warnings
from scipy.optimize import differential_evolution, LinearConstraint

# Define an empty list to store dictionaries of results
results_data = []

# Define an empty dictionary to store individual cow-level data
individual_cow_data_combined = {}

# Loop through each day
for i, smp_date in enumerate(df_81cows_91days['smp_date'].unique()): #[36:39]
    # Filter the DataFrame for the current smp_date
    current_day_df = df_81cows_91days[df_81cows_91days['smp_date'] == smp_date].reset_index(drop=True)
    # Resetting the index to start from 0 for each day


    # Calculate the total 'conc' for the current day
    total_actual_conc = current_day_df['conc'].sum()

    # Get the number of cows for the current day
    num_cows = len(current_day_df)

    # Define lists to store the predictions and objective function values
    predictions = []
    objective_values = []

    # Define objective function for optimisation
    def objective_function(x, data):
        # x corresponds to the 'conc' values for each cow
        conc_values = x

        # Update 'conc' values for current day data
        data_copy = data.copy()
        # Preprocessing steps
        data_copy["lact"] = data_copy["lact"].astype("object")
        data_copy['conc'] = conc_values

        # Drop unnecessary columns
        X = data_copy.drop(['id', 'my', 'smp_date'], axis=1)

        # Predict 'my' using the pre-trained model (loaded_model)
        predicted_my = loaded_model.predict(X)

        # Append predictions and objective function values
        predictions.append(predicted_my)
        objective_values.append(-np.sum(predicted_my))

        # Calculate total 'my' for the current day
        total_my = -np.sum(predicted_my)

        return total_my

    # Define bounds for 'conc' for each cow based on the previous day's optimal values
    if i == 0:
        # If it's the first day, use initial bounds
        bounds = [(5, 9)] * num_cows
    else:
        # Use bounds based on the previous day's optimal values
        lower_bounds = np.maximum(optimal_conc_values_day.copy() - 1, 5)  # Ensure lower bound doesn't go below 5
        upper_bounds = np.minimum(optimal_conc_values_day.copy() + 1, 9)  # Ensure upper bound doesn't go above 9
        bounds = list(zip(lower_bounds, upper_bounds)) 

    # # Define linear constraint
    linear_constraint = LinearConstraint(np.ones((1, num_cows)), lb=[total_actual_conc * 0.95], ub=[total_actual_conc])
    
    # Define linear constraint
    # linear_constraint = LinearConstraint(np.ones((1, num_cows)), lb=[-np.inf], ub=[total_actual_conc])


    # Time the optimisation process
    start_time = time.time()

    # Suppress the specific warning
    warnings.filterwarnings("ignore", message="delta_grad == 0.0.*")

    # Perform differential evolution optimisation
    result = differential_evolution(objective_function, bounds, args=(current_day_df,), constraints=[linear_constraint])

    # Re-enable warnings
    warnings.filterwarnings("default", message="delta_grad == 0.0.*")

     # Calculate the elapsed time
    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    # Print a message indicating optimisation completion for the current day
    print(f"Day {i+1}: Optimisation completed for {smp_date} in {int(hours)} hours, {int(minutes)} minutes, and {int(seconds)} seconds")

    # Get the optimal values for 'conc' for each cow
    optimal_conc_values_day = result.x

    # Sum of the optimal conc values
    total_optimal_conc = sum(optimal_conc_values_day)

    # Find the index of the iteration with the lowest objective value (maximization)
    best_iteration_index = objective_values.index(min(objective_values))

    # Get the predictions for the iteration with the lowest objective value (maximization)
    best_predictions = predictions[best_iteration_index]

    # Calculate the percentage increase in Milk yield
    percentage_increase_my = ((-min(objective_values) - current_day_df['my'].sum()) / current_day_df['my'].sum()) * 100
 
    # Append the results for the current day to the results data list
    results_data.append({'smp_date': smp_date,
                         'Num_Cows': num_cows,
                         'Total_Actual_Conc': total_actual_conc,
                         'Total_Actual_MY': current_day_df['my'].sum(),
                         'Total_Optimal_Conc': total_optimal_conc,
                         'Objective_Function': min(objective_values),
                         'Percentage_Increase_MY': percentage_increase_my,
                         'Iterations': result.nit,
                         'Max_MY_Iteration': best_iteration_index,
                         'Optimisation_Time': elapsed_time})

    # Store individual cow-level data in the dictionary
    for index, row in current_day_df.iterrows():
        cow_id = row['id']
        actual_yield = row['my']
        actual_conc = row['conc']
        predicted_yield = best_predictions[index]
        optimal_concentrate = optimal_conc_values_day[index]

        individual_cow_data_combined.setdefault(smp_date, {}).setdefault(cow_id, []).append({
            'Predicted Milk Yield': predicted_yield,
            'Optimal Concentrate': optimal_concentrate,
            'Actual Concentrate': actual_conc,
            'Actual Milk Yield': actual_yield
        })

# Convert the results data list to a DataFrame
results_df_combined = pd.DataFrame(results_data)

# Flatten individual_cow_data to a list of records
records = []
for date, cows in individual_cow_data_combined.items():
    for cow_id, data_list in cows.items():
        for data in data_list:
            record = data.copy()
            record['Date'] = date
            record['Cow ID'] = cow_id
            records.append(record)

# Create a DataFrame
df_opt = pd.DataFrame(records)

# Reorder the columns
df_opt = df_opt[['Date', 'Cow ID', 'Predicted Milk Yield', 'Optimal Concentrate', 'Actual Concentrate', 'Actual Milk Yield']]

# Save to CSV
csv_file_results = 'results_df_combined.csv'
csv_file_individual = 'individual_cow_data_combined.csv'
results_df_combined.to_csv(csv_file_results, index=False)
df_opt.to_csv(csv_file_individual, index=False)
print(results_df_combined)
print(results_df_combined["Percentage_Increase_MY"].mean())



# ### The End
