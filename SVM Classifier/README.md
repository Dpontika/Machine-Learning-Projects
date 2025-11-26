# **SVM with RBF Kernel**



This exercise implements an SVM classifier that learns to distinguish between Iris-versicolor (class 1) and other iris species (class 0) using the Gaussian RBF kernel. 

SVM with RBF kernel can capture complex patterns by mapping data to higher-dimensional spaces.



## **Features**



&nbsp;   Kernel Method: Uses RBF kernel for non-linear classification



&nbsp;   Parameter Tuning: User-defined gamma and C parameters



&nbsp;   Cross-Validation: 9-fold validation for reliable performance estimation



&nbsp;   Comprehensive Evaluation: 6 performance metrics 



&nbsp;   Visualization: Side-by-side comparison of predictions vs actual values





## **Core Algorithm** 



#### 1\. SVM with RBF Kernel Training



&nbsp;   Method: Support Vector Machine with Radial Basis Function kernel



&nbsp;   Kernel Formula: k(x, y) = exp(-gamma \* ||x - y||²)



&nbsp;   Objective: Find optimal hyperplane that maximizes margin between classes



&nbsp;   Output: Support vectors and decision function parameters



&nbsp;   Key Parameters:



&nbsp;   	Gamma (γ): Kernel coefficient controlling influence range



&nbsp;       	Small gamma: Far influence, smoother decision boundaries



&nbsp;       	Large gamma: Close influence, more complex boundaries



&nbsp;   	C: Regularization parameter controlling error tolerance



&nbsp;       	Small C: More tolerant of classification errors



&nbsp;       	Large C: Strict, tries to classify every point correctly



#### 2. Prediction



&nbsp;   Method: Kernel-based decision function



&nbsp;   Process: Maps input features to high-dimensional space and applies linear separation



&nbsp;   Output: Binary classification (0/1) based on decision function sign



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



