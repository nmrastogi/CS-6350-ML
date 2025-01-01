import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(42)
class Perceptron:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.w = None
        self.b = None
        self.total_updates = 0
        self.dev_accuracies = []
    def train(self, X, y, epochs):
        n_samples, n_features = X.shape
        # Initialize weights and bias
        self.w = np.random.uniform(-0.01, 0.01, size=n_features)
        self.b = np.random.uniform(-0.01, 0.01)
        best_dev_accuracy = 0
        best_perceptron = None

        # Training loop
        for _ in range(epochs):
            for i in range(n_samples):
                if y[i] * (np.dot(X[i], self.w) + self.b) <= 0:
                    # Update weights and bias
                    self.w += self.learning_rate * y[i] * X[i]
                    self.b += self.learning_rate * y[i]
                    self.total_updates += 1

                dev_accuracy = evaluate(X_dev, y_dev, self)
                self.dev_accuracies.append(dev_accuracy)
            # Check if this epoch achieves the highest development accuracy
                if dev_accuracy > best_dev_accuracy:
                    best_dev_accuracy = dev_accuracy
                    best_perceptron = Perceptron(self.learning_rate)
                    best_perceptron.w = self.w.copy()
                    best_perceptron.b = self.b

        return best_perceptron

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

def cross_validation(learning_rates, epochs_cv):
    results = []
    for eta in learning_rates:
        fold_accuracies = []
        #dev_fold_accuracies = []
        for i in range(5):
            # Load validation data for current fold
            val_data = pd.read_csv(f"CVSplits/train{i}.csv", header=None,skiprows=1)
            y_val_fold = val_data.iloc[:, 0].values.astype(int)
            X_val_fold = val_data.iloc[:, 1:].values.astype(float)

            # Training and evaluating the model
            perceptron = Perceptron(eta)
            best_perceptron = perceptron.train(X_train, y_train, epochs_cv)
            accuracy = evaluate(X_val_fold, y_val_fold, best_perceptron)
            fold_accuracies.append(accuracy)
            

        # Average accuracy for this learning rate
        avg_accuracy = np.mean(fold_accuracies)
        results.append((eta, avg_accuracy))

    # Choose the best learning rate
    best_eta, best_accuracy = max(results, key=lambda x: x[1])
    return best_eta, best_accuracy

def evaluate(X, y, perceptron):
    y_pred = perceptron.predict(X)
    accuracy = np.mean(y_pred == y)
    return accuracy

# Load data
train_data = pd.read_csv("diabetes.train.csv", header=None,skiprows=1)
X_train = train_data.iloc[:, 1:].values.astype(float)
y_train = train_data.iloc[:, 0].values.astype(int)

dev_data = pd.read_csv("diabetes.dev.csv", header=None,skiprows=1)
X_dev = dev_data.iloc[:, 1:].values.astype(float)
y_dev = dev_data.iloc[:, 0].values.astype(int)

# Define hyperparameters
learning_rates = [1, 0.1, 0.01]
epochs_cv = 10

# Cross-validation on development set to find the best learning rate
best_eta, best_accuracy = cross_validation(learning_rates, epochs_cv)
print("Simple perceptron")
print("Best learning rate found:", best_eta)
print("Best accuracy: ", best_accuracy)

# Train the model with the best learning rate for more epochs
epochs_train = 20
perceptron = Perceptron(best_eta)
best_perceptron = perceptron.train(X_train, y_train, epochs_train)
print("Total number of updates on the training set:", perceptron.total_updates)
#dev_accuracies = perceptron.train(X_train, y_train, epochs=20)

# Evaluate the best-performing perceptron on the test set
test_data = pd.read_csv("diabetes.test.csv", header=None,skiprows=1)
X_test = test_data.iloc[:, 1:].values.astype(float)
y_test = test_data.iloc[:, 0].values.astype(int)
test_accuracy = evaluate(X_test, y_test, best_perceptron)
print("Test accuracy using the best-performing perceptron:", test_accuracy)

dev_accuracy = evaluate(X_dev, y_dev, perceptron)
print("Dev accuracy:", dev_accuracy)
#Most frequent label
most_frequent_label = train_data.iloc[:, 0].mode()[0]

# Calculate accuracy on the test set
test_accuracy = (test_data.iloc[:, 0] == most_frequent_label).mean()

# Calculate accuracy on the development set
dev_accuracy = (dev_data.iloc[:, 0] == most_frequent_label).mean()

print("Accuracy on test set (always predicting most frequent label):", test_accuracy)
print("Accuracy on development set (always predicting most frequent label):", dev_accuracy)

# plt.figure(figsize=(10, 6))
# plt.plot(range(1, len(dev_accuracies) + 1), dev_accuracies, marker='o', linestyle='-')
# plt.title("Learning Curve (Development Set Accuracy)")
# plt.xlabel("Epoch")
# plt.ylabel("Dev Set Accuracy")
# plt.xticks(range(1, len(dev_accuracies) + 1))
# plt.grid(True)
# plt.show()