# Machine Learning - Cross Validation
# @author: Dimitris Pontikakis

# Import  libraries 
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.model_selection import train_test_split

# Load the dataset
data = read_csv('iris.data', header=None).values

# Number of attributes and patterns
number_of_patterns, number_of_attributes = data.shape
print(f"Number of attributes: {number_of_attributes}")
print(f"Number of paterns: {number_of_patterns}")

# One vs Rest multiclass classification
# Create dictionary map_dict
map_dict = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 0}

# Select features: first 4 columns (sepal length, sepal width, petal length, petal width)
x = data[:, 0:4]
print(f"Features shape: {x.shape}")

# Select targets: initialize with zeros, then fill based on class names
labels = data[:, 4]
t = np.zeros([number_of_patterns], dtype=int)

for pattern in range(number_of_patterns):
    t[pattern] = map_dict[labels[pattern]]
    
print(f"Targets shape: {t.shape}")
print(f"Target values: {np.unique(t)}")

# Cross-validation with 9 folds
print('9-Fold Cross Validation - Training (Blue) vs Testing (Red) Data')

for fold in range(9):
    # Split data into training and testing sets
    xtrain, xtest, ttrain, ttest = train_test_split(x, t, test_size=0.1)
        
    plt.subplot(3, 3, fold + 1)
    plt.plot(xtrain[:, 0], xtrain[:, 2], "b.")
    plt.plot(xtest[:, 0], xtest[:, 2], "r.")
    plt.xlabel('Sepal Length (cm)')      # X-axis label
    plt.ylabel('Petal Length (cm)')      # Y-axis label
    plt.title('Fold ' + str(fold))

plt.tight_layout()
plt.show()