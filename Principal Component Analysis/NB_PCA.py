# Feature Selection with PCA
# @Author: Dimitris Pontikakis

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA

# Load and prepare the dataset
data = np.load('mnist_49.npz')
x = data['x']
t = data['t']

number_of_patterns, number_of_features = x.shape
print(f"Dataset loaded: {number_of_patterns} samples, {number_of_features} features")
print(f"Digit distribution: {np.sum(t == 0)} samples of digit '4', {np.sum(t == 1)} samples of digit '9'")

print("\n=== NAÏVE BAYES ON ORIGINAL DATA ===")
score_train = 0.0
score_test = 0.0

# Cross-validation with 10 folds
for i in range(10):
    xtrain, xtest, ttrain, ttest = train_test_split(x, t, test_size=0.1)

    model = GaussianNB()
    model.fit(xtrain, ttrain)

    score_train += model.score(xtrain, ttrain)
    score_test += model.score(xtest, ttest)
    
print("Mean Train accuracy: {}".format(score_test / 10))
print("Mean Test accuracy: {}".format(score_train / 10))


print("\n=== PCA WITH DIFFERENT COMPONENTS ===")
num_components = [1, 2, 5, 10, 20, 30, 40, 50, 100, 200]

# Store accuracies
acc_train = []
acc_test = []

for num in num_components:
    print(f"\nTesting with {num} PCA components...")
    pca = PCA(n_components=num)
    x_pca = pca.fit_transform(x) # Transform data to PCA space

    # Initialize accumulators for this component count
    mean_test_score = 0.0
    mean_train_score = 0.0

    # 10-fold cross-validation with PCA-transformed data
    for i in range(10):
        xtrain, xtest, ttrain, ttest = train_test_split(x_pca, t, test_size=0.1)

        model = GaussianNB()
        model.fit(xtrain, ttrain)

        mean_test_score += model.score(xtest, ttest)
        mean_train_score += model.score(xtrain, ttrain)

    # Store accuracies for plotting
    acc_test.append(mean_test_score / 10)
    acc_train.append(mean_train_score / 10)
    
    print(f"  Training Accuracy: {mean_train_score/10:.4f}")
    print(f"  Testing Accuracy: {mean_test_score/10:.4f}")

plt.figure(figsize=(10, 6))
plt.subplot(1, 1, 1)

# Trainning accuracy curve
plt.plot(num_components, acc_train, 'b-o', linewidth=2, markersize=8, label='Training Accuracy')

# Testing accuracy curve
plt.plot(num_components, acc_test, 'r-s', linewidth=2, markersize=8, label='Testing Accuracy')

plt.xlabel('Number of PCA Components', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('PCA Feature Selection: Accuracy vs Number of Components', fontsize=14)
plt.xscale('log') # Log scale for x-axis to better visualize small component counts
plt.xticks(num_components, [str(n) for n in num_components])
plt.grid(True, alpha=0.3)
plt.legend(loc='best')
plt.tight_layout()
plt.show()

print("\n=== OBSERVATIONS ===")
# Find best number of components
best_test_idx = np.argmax(acc_test)
best_components = num_components[best_test_idx]
best_test_accuracy = acc_test[best_test_idx]
print(f"Best number of PCA components: {best_components}")
print(f"Best testing accuracy with PCA: {best_test_accuracy:.4f}")
print(f"Testing accuracy without PCA: {score_test /10:.4f}")
