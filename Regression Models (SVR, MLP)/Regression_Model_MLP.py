# Machine Learning - MLP Regression
# @Author: Dimitris Pontikakis

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

def regevaluate(t, predict, criterion):
    """
    Evaluate regression performance
    
    Parameters:
    t: array of actual target values
    predict: array of predicted values
    criterion: string specifying evaluation metric ('mse' or 'mae')
    
    Returns:
    Calculated metric value
    """
    
    if criterion == 'mse':
        return mean_squared_error(t, predict)
    else:
        return mean_absolute_error(t, predict)


def regrevaluate(t, predict, criterion):
    """
    Custom implementation of regression evaluation metrics
    
    Parameters:
    t: array of actual target values
    predict: array of predicted values  
    criterion: string specifying evaluation metric ('mse' or 'mae')
    
    Returns:
    Calculated metric value
    """
    # Initialize arrays to store differences
    difference = np.zeros(len(predict))
    squared_dif = np.zeros(len(predict))

    value = 0
    
    # Calculate differences between actual and predicted values
    for i in range(len(predict)):
        difference[i] = t[i] - predict[i]
        squared_dif[i] = difference[i] ** 2

    if criterion == 'mse':
        value = 1 / len(predict) * sum(squared_dif)

    if criterion == 'mae':
        value = 1 / abs(len(predict) * sum(difference))

    return value
    
# Load and prepare the dataset
data = pd.read_csv('housing.data', header=None, sep="\s+").values

number_of_patterns, number_of_attributes = data.shape
print(f"Dataset loaded: {number_of_patterns} samples, {number_of_attributes} attributes")

x = data[:, 0:number_of_attributes - 1] # First 13 columns are features
t = data[:, number_of_attributes - 1] # Last column is the target (MEDV)

print(f"Features shape: {x.shape}")
print(f"Targets shape: {t.shape}")

lowest_mean_mse = 1000.0    # Initialize with high value to find minimum
lowest_mean_mae = 1000.0    # Initialize with high value to find minimum

lowest_n_mae = 0.0
lowest_n_mse = 0.0

# Define MLP parameters
max_epoch = 100000
activation_function = 'relu'
solver_method = 'lbfgs'
lr = 0.001

print("\n=== MLP CONFIGURATION ===")  
print(f"Epochs: {max_epoch}")
print('Activation function: {}'.format(activation_function))
print('Solver: {}'.format(solver_method))
print('Learning rate: {}'.format(lr))

print("\n=== SEARCH FOR BEST MLP PARAMETERS ===")

# Test different parameter combinations
for n in [5, 10, 20, 30, 40, 50]:
    mean_mae = 0.0
    mean_mse = 0.0

    # Cross-validation with 9 folds
    for fold in range(9):
        xtrain, xtest, ttrain, ttest = train_test_split(x, t, test_size=0.1)
        
        # Create MLP  
        mlp = MLPRegressor(
        hidden_layer_sizes=(n,),
        activation=activation_function,
        solver=solver_method,
        learning_rate='constant',
        learning_rate_init=lr,
        max_iter=max_epoch,)
        
        # Train the MLP
        mlp.fit(xtrain, ttrain)
    
        # Make predictions
        y_pred = mlp.predict(xtest)
        predict = y_pred
    
        mean_mae += regevaluate(ttest, predict, 'mae')
        mean_mse += regevaluate(ttest, predict, 'mse')
        
    # Average metrics accross 9 folds
    mean_mae /= 9
    mean_mse /= 9
    
    if lowest_mean_mae > mean_mae:
        lowest_mean_mae = mean_mae
        lowest_n_mae = n

    if lowest_mean_mse > mean_mse:
        lowest_mean_mse = mean_mse
        lowest_n_mse = n
    
    print(f"\nN={n:<4} | MSE: {mean_mse:>7.4f} | MAE: {mean_mae:>6.4f}")

print("\n=== BEST PARAMETERS ===")
print("Mean MAE: {}, Hidden Neurons: {}".format(lowest_mean_mae, lowest_n_mae))
print("Mean MSE: {}, Hidden Neurons: {}".format(lowest_mean_mse, lowest_n_mse))

# Create final model with best parameters
xtrain, xtest, ttrain, ttest = train_test_split(x, t, test_size=0.1)
model  = MLPRegressor(
        hidden_layer_sizes=(lowest_n_mse,),
        activation=activation_function,
        solver=solver_method,
        learning_rate='constant',
        learning_rate_init=lr,
        max_iter=max_epoch,

    )
model .fit(xtrain, ttrain)
predict = model.predict(xtest)

plt.figure(figsize=(12, 6))
plt.plot(ttest, 'b-', label='Actual Values', linewidth=2, markersize=6) # Blue line for target values
plt.plot(predict, 'ro', label='Predicted Values', markersize=6) # Red circles for MLP predictions
plt.title(f'MLP Regression: Actual vs Predicted Values')
plt.xlabel('Sample Index')
plt.ylabel('MEDV (House Price in $1000s)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

