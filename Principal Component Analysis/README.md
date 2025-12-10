# **Feature Selection with Principal Component Analysis (PCA)**



This project implements PCA for feature selection and dimensionality reduction in the context of digit classification. Using the MNIST 4 vs 9 dataset, it's demonstrated how PCA transforms 784 raw pixel features into a smaller set of meaningful components and significantly improves the performance of a Gaussian Naïve Bayes classifier.



## **Features**



    Dimensionality Reduction: Transforms 784-pixel images into compact feature representations



    Performance Optimization: Identifies optimal number of components for maximum accuracy



    Cross-Validation: 10-fold validation for reliable performance estimation



    Visual Analytics: Clear visualization of accuracy vs components trade-off



## **Dataset**



    Source: MNIST 4 vs 9 subset



    Samples: 11,791 handwritten digit images



    Classes: Binary classification



        Class 0: Digit "4" (5,842 samples)



        Class 1: Digit "9" (5,949 samples)



    Original Features: 784 pixels (28×28 grayscale images)





## **Core Algorithm**



1. #### Gaussian Naïve Bayes with PCA



Training Process:



    Apply PCA to training data



    Train Gaussian Naïve Bayes on reduced features



    Estimate Gaussian parameters for each class



Prediction Process:



    Transform input using PCA



    Calculate class probabilities using Bayes' theorem



    Assign class with highest probability





#### 2\. Performance Analysis



Key Results



    Without PCA: 72.5% accuracy



    With PCA (20 components): 94.0% accuracy



    Improvement: +21.5% better with PCA



Optimal Configuration



    Best components: 20



    Variance captured: 69.3%



    Compression ratio: 784 → 20 (39:1 reduction)



## **Observations**



    PCA Improves Performance: 20 PCA features outperform 784 raw pixels



    Quality Over Quantity: 69% variance gives better accuracy than 97% variance



    Sweet Spot Exists: 20-50 components optimal for this task



    Less Can Be More: Extreme compression (1-5 components) underfits



    Noise Matters: Including too many components reduces accuracy



    Faster Training: 39x fewer features = much faster computation



    Storage Savings: 96% less storage needed

