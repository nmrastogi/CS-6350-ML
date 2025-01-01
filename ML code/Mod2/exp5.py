import numpy as np
import pandas as pd
np.random.seed(42)
class AggressivePerceptron:
    def __init__(self, margin):
        self.margin = margin
        self.w = None
        self.total_updates = 0

    def train(self, X, y, epochs):
        n_samples, n_features = X.shape
        # Initialize weights
        self.w = np.zeros(n_features)

        # Training loop
        for _ in range(epochs):
            for i in range(n_samples):
                if y[i] * np.dot(X[i], self.w) <= self.margin:
                    # Compute the learning rate
                    learning_rate = (self.margin - y[i] * np.dot(X[i], self.w)) / (np.dot(X[i], X[i]) + 1)
                    # Update weights
                    self.w += learning_rate * y[i] * X[i]
                    self.total_updates += 1 

    def predict(self, X):
        return np.sign(np.dot(X, self.w))

def cross_validation(X_train, y_train, margins, epochs_cv):
    results = []
    for margin in margins:
        fold_accuracies = []
        for i in range(5):
            # Splitting data into train and validation sets
            val_data = pd.read_csv(f"CVSplits/train{i}.csv", header=None,skiprows=1)
            y_val_fold = val_data.iloc[:, 0].values.astype(int)
            X_val_fold = val_data.iloc[:, 1:].values.astype(float)

            # Training and evaluating the model
            perceptron = AggressivePerceptron(margin)
            perceptron.train(X_train, y_train, epochs_cv)
            accuracy = evaluate(X_val_fold, y_val_fold, perceptron)
            fold_accuracies.append(accuracy)

        # Average accuracy for this margin
        avg_accuracy = np.mean(fold_accuracies)
        results.append((margin, avg_accuracy))

    # Choose the best margin
    best_margin, best_accuracy = max(results, key=lambda x: x[1])
    return best_margin, best_accuracy

def evaluate(X, y, perceptron):
    y_pred = perceptron.predict(X)
    accuracy = np.mean(y_pred == y)
    return accuracy

# Load data
train_data = pd.read_csv("diabetes.train.csv", header=0)
y_train = train_data.iloc[:, 0].values.astype(int)
X_train = train_data.iloc[:, 1:].values.astype(float)

# Define hyperparameters
margins = [1, 0.1, 0.01]
epochs_cv = 10

dev_data = pd.read_csv("diabetes.dev.csv", header=0)
y_dev = dev_data.iloc[:, 0].values.astype(int)
X_dev = dev_data.iloc[:, 1:].values.astype(float)

# Perform cross-validation to choose the best margin
best_margin, best_accuracy = cross_validation(X_train, y_train, margins, epochs_cv)

# Train the model with the best margin for 20 epochs
epochs_train = 20
perceptron = AggressivePerceptron(best_margin)
perceptron.train(X_train, y_train, epochs_train)
print("Aggresive perceptron")
print("Best margin found:", best_margin)
print("Best accuracy found:", best_accuracy)
print("Total number of updates during training:", perceptron.total_updates)
# Load test data
test_data = pd.read_csv("diabetes.test.csv", header=0)
y_test = test_data.iloc[:, 0].values.astype(int)
X_test = test_data.iloc[:, 1:].values.astype(float)

# Evaluate accuracy on test set
test_accuracy = evaluate(X_test, y_test, perceptron)
print("Accuracy on test set:", test_accuracy)

dev_accuracy = evaluate(X_dev, y_dev, perceptron)
print("Dev accuracy:", dev_accuracy)