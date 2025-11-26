# **Perceptron Classifier with Cross-Validation**



This exercise implements a Perceptron neural network that learns to distinguish between Iris-versicolor (class 1) and other iris species (class 0). 

The Perceptron learns iteratively through error correction.



## **Features**



&nbsp;   Online Learning: Iterative weight updates based on prediction errors



&nbsp;   Cross-Validation: 9-fold validation for reliable performance estimation



&nbsp;   Comprehensive Evaluation: 6 performance metrics



&nbsp;   Visualization: Side-by-side comparison of predictions vs actual values





## **Core Algorithm - Perceptron Learning**



#### Training Process:



For each epoch up to max\_epoch:

&nbsp;   For each training sample:

&nbsp;       Calculate activation: u = x · w

&nbsp;       Make prediction: y = 1 if u ≥ 0, else 0

&nbsp;       If prediction wrong: update weights: w = w + β(t - y)x

&nbsp;   Stop early if no errors made



#### Key Parameters:



&nbsp;   max\_epoch: Maximum training iterations (user input)



&nbsp;   beta: Learning rate - controls update size (user input)



#### Data Flow



&nbsp;   Data Loading: Iris dataset with binary classification



&nbsp;   Feature Augmentation: Add bias term column of 1s



&nbsp;   Training: Iterative weight updates on 90% of data



&nbsp;   Testing: Prediction on remaining 10% of data



&nbsp;   Evaluation: Calculate 6 performance metrics



&nbsp;   Visualization: 3×3 plot grid showing all folds



#### Performance Metrics



    Accuracy



    Formula: (TP + TN) / (TP + TN + FP + FN)



    Interpretation: Overall classification correctness

 



    Precision



    Formula: TP / (TP + FP)



    Interpretation: How many predicted positives are actually positive





    Recall



    Formula: TP / (TP + FN)



    Interpretation: How many actual positives are correctly identified





    F-Measure



    Formula: (precision x recall) / (precision + recall) / 2



    Interpretation: Harmonic mean of precision and recall





    Sensitivity



    Formula: Same as recall



    Interpretation: Ability to detect positive cases





    Specificity



    Formula: TN / (TN + FP)



    Interpretation: Ability to detect negative cases

