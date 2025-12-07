# **Naïve Bayes Classifier with Gaussian Distribution**



This project implements a Naïve Bayes classifier that distinguishes between Iris-versicolor (class 1) and other iris species (class 0) using Gaussian distribution assumptions for feature probabilities. 

The classifier follows Bayes' Theorem with the "naïve" assumption that all features are conditionally independent given the class label.



## **Features**



&nbsp;   Probabilistic Classification: Uses Bayes' Theorem to calculate class probabilities



&nbsp;   Gaussian Assumption: Models continuous features with normal distributions



&nbsp;   Cross-Validation: 9-fold validation for reliable performance estimation



&nbsp;   Comprehensive Evaluation: 6 performance metrics



&nbsp;   Visualization: Side-by-side comparison of predictions vs actual values



&nbsp;   Custom Implementation: Manual training and prediction functions for educational purposes





## **Core Algorithm**



#### 1\. Naïve Bayes Training



Method: Maximum Likelihood Estimation with Gaussian distributions



Training Process:



&nbsp;   Calculate Prior Probabilities: P(Class 0) and P(Class 1) based on class frequencies



&nbsp;   Estimate Gaussian Parameters: For each class and each feature:



&nbsp;       Mean (μ): Average value of the feature for that class



&nbsp;       Standard Deviation (σ): Spread of the feature values for that class



&nbsp;   Store Model: Create dictionary containing priors, means, and standard deviations



Mathematical Foundation:



&nbsp;   Prior Probability: P(C) = (# samples in class C) / (total samples)



&nbsp;   Likelihood (Gaussian): P(xᵢ|C) = (1/√(2πσ²)) \* exp(-(xᵢ-μ)²/(2σ²))



&nbsp;   Naïve Assumption: P(x₁,x₂,...,xₙ|C) = ∏ P(xᵢ|C) (features independent given class)



#### 2\. Prediction



Method: Bayesian inference with likelihood ratios



Prediction Process:



&nbsp;   Calculate Prior Ratio: P(Class 1) / P(Class 0)



&nbsp;   Update with Likelihoods: Multiply by product of Gaussian PDF ratios for each feature



&nbsp;   Decision Rule:



&nbsp;       If likelihood ratio < 1: Predict Class 0



&nbsp;       If likelihood ratio > 1: Predict Class 1



Bayes' Theorem Application:

P(Class 1 | Data) / P(Class 0 | Data) = \[P(Class 1)/P(Class 0)] × ∏ \[P(Featureᵢ | Class 1)/P(Featureᵢ | Class 0)]



#### 3\. Performance Metrics



Accuracy



&nbsp;   Formula: (TP + TN) / (TP + TN + FP + FN)



&nbsp;   Interpretation: Overall classification correctness rate



Precision



&nbsp;   Formula: TP / (TP + FP)



&nbsp;   Interpretation: Proportion of positive predictions that are correct



&nbsp;   Application: Important when false positives are costly



Recall (Sensitivity)



&nbsp;   Formula: TP / (TP + FN)



&nbsp;   Interpretation: Proportion of actual positives correctly identified



&nbsp;   Application: Important when false negatives are costly



F-Measure (F1 Score)



&nbsp;   Formula: (2 × Precision × Recall) / (Precision + Recall)



&nbsp;   Interpretation: Harmonic mean balancing precision and recall



&nbsp;   Application: Single metric for imbalanced class performance



Sensitivity



&nbsp;   Formula: Same as Recall (TP / (TP + FN))



&nbsp;   Interpretation: Ability to detect positive cases



&nbsp;   Application: Medical testing, fraud detection



Specificity



&nbsp;   Formula: TN / (TN + FP)



&nbsp;   Interpretation: Ability to detect negative cases



&nbsp;   Application: Quality control, spam filtering

