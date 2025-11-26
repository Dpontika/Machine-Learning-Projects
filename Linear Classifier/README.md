# **Linear Classifier with Cross-Validation**



This exercise implements a linear binary classifier using the Iris dataset. The program builds a linear classifier that distinguishes between Iris-versicolor (class 1) and other iris species (class 0).

Using mathematical optimization through pseudo-inverse, the classifier learns optimal weights to separate the two classes. Performance is rigorously evaluated across multiple data splits to ensure reliable results.



## **Features**



    Data Preprocessing: Binary classification setup and feature augmentation



    Linear Classification: Optimal weight calculation using pseudo-inverse method



    Cross-Validation: 9-fold validation for performance estimation



    Comprehensive Evaluation: 6 different performance metrics



    Visualization: Side-by-side comparison of predictions vs actual values





## **Core Algorithms**



#### 1\. Linear Classifier Training



    Method: Pseudo-inverse (Moore-Penrose inverse)



    Formula: w = X⁺ · t where X⁺ is pseudo-inverse of feature matrix



    Output: Weight vector w of length 5 (4 features + 1 bias)





#### 2\. Prediction



    Decision Boundary: Threshold at 0



    y ≥ 0 → predict class 1



    y < 0 → predict class 0



#### 3\. Performance Metrics



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



## **Output**

Mean accuracy: 0.7037037037037037

Mean precision: 0.7235449735449735

Mean recall: 0.46825396825396826

Mean fmeasure: 0.5359147025813692

Mean sensitivity: 0.46825396825396826

Mean specificity: 0.8621773288439957



