# **Regression with SVR and MLP - Updated Results**



## **SVR Regression (Support Vector Regression)**



This implementation uses Support Vector Regression with RBF kernel to predict housing prices from the Boston Housing dataset. 

SVR finds a function that deviates from actual training data by a value no greater than epsilon while being as flat as possible.



### Features



&nbsp;   Kernel Method: Uses RBF kernel for non-linear regression



&nbsp;   Parameter Tuning: Grid search over gamma and C parameters



&nbsp;   Cross-Validation: 9-fold validation for reliable performance estimation



&nbsp;   Performance Metrics: Mean Squared Error (MSE) and Mean Absolute Error (MAE)



&nbsp;   Visualization: Comparison of predicted vs actual housing prices



### Core Algorithm



#### 1\. SVR with RBF Kernel Training



&nbsp;   Method: Support Vector Regression with Radial Basis Function kernel



&nbsp;   Kernel Formula: k(x, y) = exp(-gamma \* ||x - y||²)



&nbsp;   Objective: Find optimal regression function with epsilon-insensitive loss



&nbsp;   Key Parameters:



&nbsp;       Gamma (γ): Kernel coefficient controlling influence range



&nbsp;           Small gamma: Far influence, smoother regression function



&nbsp;           Large gamma: Close influence, more complex regression function



&nbsp;       C: Regularization parameter controlling error tolerance



&nbsp;           Small C: More tolerant of prediction errors



&nbsp;           Large C: Strict, tries to fit training data more closely



#### 2\. Performance Metrics



&nbsp;   Mean Squared Error (MSE)



&nbsp;       Formula: (1/n) \* Σ(actual - predicted)²



&nbsp;       Interpretation: Penalizes larger errors more heavily



&nbsp;   Mean Absolute Error (MAE)



&nbsp;       Formula: (1/n) \* Σ|actual - predicted|



&nbsp;       Interpretation: Average magnitude of errors



### Output Analysis



=== SEARCH FOR BEST SVR PARAMETERS ===



γ=0.0001  C=1    | MSE: 62.1630 | MAE: 5.0383

γ=0.0001  C=10   | MSE: 51.9731 | MAE: 4.2886

γ=0.0001  C=100  | MSE: 27.3944 | MAE: 3.3330

γ=0.0001  C=1000 | MSE: 16.8581 | MAE: 2.6639



γ=0.001   C=1    | MSE: 61.8186 | MAE: 4.9868

γ=0.001   C=10   | MSE: 42.2968 | MAE: 4.0886

γ=0.001   C=100  | MSE: 24.6877 | MAE: 3.5875

γ=0.001   C=1000 | MSE: 38.7323 | MAE: 4.3065



γ=0.01    C=1    | MSE: 72.6296 | MAE: 5.7210

γ=0.01    C=10   | MSE: 47.3246 | MAE: 4.5491

γ=0.01    C=100  | MSE: 43.0399 | MAE: 4.6507

γ=0.01    C=1000 | MSE: 42.2844 | MAE: 4.4390



γ=0.1     C=1    | MSE: 105.3498 | MAE: 7.3632

γ=0.1     C=10   | MSE: 73.2801 | MAE: 5.9263

γ=0.1     C=100  | MSE: 75.0639 | MAE: 6.1340

γ=0.1     C=1000 | MSE: 70.4615 | MAE: 5.9114



=== BEST PARAMETERS ===

Mean MAE: 2.6639, gamma: 0.0001, C: 1000

Mean MSE: 16.8581, gamma: 0.0001, C: 1000



#### Interpretation:



&nbsp;   Best Performance: Achieved with gamma=0.0001 and C=1000 for both MSE and MAE



&nbsp;   Parameter Trends:



&nbsp;       For gamma=0.0001, increasing C consistently improves performance



&nbsp;       Higher gamma values (0.01, 0.1) perform poorly, suggesting overfitting



&nbsp;       The optimal configuration indicates a smooth regression function with strong regularization



&nbsp;   Error Metrics:



&nbsp;       MSE of 16.86 means average squared error of ~$4,100 (since units are $1000s)



&nbsp;       MAE of 2.66 means average prediction error of ~$2,660



## **MLP Regression (Multi-Layer Perceptron)**



This implementation uses a Neural Network to predict housing prices. The MLP learns complex non-linear relationships between housing features and prices through backpropagation. 



### Features



&nbsp;   Activation Function: ReLU for hidden layer



&nbsp;   Optimization: L-BFGS solver for efficient training



&nbsp;   Cross-Validation: 9-fold validation for performance estimation



&nbsp;   Hyperparameter Tuning: Search over hidden layer sizes \[5, 10, 20, 30, 40, 50]



### Core Algorithm



#### 1\. MLP Architecture



&nbsp;   Input Layer: 13 neurons (housing features)



&nbsp;   Hidden Layer: Variable neurons with ReLU activation



&nbsp;   Output Layer: 1 neuron (housing price prediction)



&nbsp;   Training: Backpropagation with L-BFGS optimization



#### 2\. Key Parameters



&nbsp;   Hidden Layer Size: Number of neurons in hidden layer



&nbsp;       Too few: Underfitting, cannot capture complexity



&nbsp;       Too many: Overfitting, memorizes training data



&nbsp;   Activation Function: ReLU (Rectified Linear Unit)



&nbsp;   Solver: L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno)



&nbsp;   Learning Rate: 0.001 (constant throughout training)



&nbsp;   Max Iterations: 100,000 epochs 



## Output Analysis text



=== MLP CONFIGURATION ===

Epochs: 100000

Activation function: relu

Solver: lbfgs

Learning rate: 0.001



=== SEARCH FOR BEST MLP PARAMETERS ===



N=5    | MSE: 35.5119 | MAE: 4.2022

N=10   | MSE: 24.0679 | MAE: 3.3819

N=20   | MSE: 18.1105 | MAE: 3.0031

N=30   | MSE: 15.6921 | MAE: 2.7902

N=40   | MSE: 13.2477 | MAE: 2.4917

N=50   | MSE: 14.7556 | MAE: 2.6554



=== BEST PARAMETERS ===

Mean MAE: 2.4917, Hidden Neurons: 40

Mean MSE: 13.2477, Hidden Neurons: 40



#### Interpretation:



&nbsp;   Performance Trends:



&nbsp;	Small networks (N=5,10) underperform due to limited capacity



&nbsp;       Clear optimal point at N=40 neurons for both MSE and MAE



&nbsp;       Performance degrades slightly at N=50, suggesting potential overfitting



&nbsp;   Best Performance:



&nbsp;       MSE: 13.2477 with N=40 neurons 



&nbsp;       MAE: 2.4917 with N=40 neurons



&nbsp;   Error Metrics:



&nbsp;       Best MSE of 13.25 corresponds to average squared error of ~$3,640



&nbsp;       Best MAE of 2.49 means average prediction error of ~$2,490



# Model Comparison 

Performance Summary



&nbsp;   SVR Best: MSE=16.8581, MAE=2.6639



&nbsp;   MLP Best: MSE=13.2477, MAE=2.4917



Key Insights 



&nbsp;   MLP Superiority: With enhanced training (100,000 epochs), MLP significantly outperforms SVR



&nbsp;   Performance Gains: MLP shows 21.4% better MSE and 9.7% better MAE than SVR



&nbsp;   Optimal Architecture: 40 hidden neurons provide the best balance of complexity and generalization







