import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

        # Training loop
        for epoch in range(epochs):
            for i in range(n_samples):
                if y[i] * (np.dot(X[i], self.w) + self.b) <= 0:
                    # Update weights and bias
                    self.w += self.learning_rate * y[i] * X[i]
                    self.b += self.learning_rate * y[i]
                    self.total_updates += 1

            # Evaluate on development set at the end of each epoch
            dev_accuracy = evaluate(X_dev, y_dev, self)
            self.dev_accuracies.append(dev_accuracy)

        return self.dev_accuracies

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

def cross_validation(learning_rates, epochs_cv):
    results = []
    for eta in learning_rates:
        fold_accuracies = []
        for i in range(5):
            # Load validation data for current fold
            val_data = pd.read_csv(f"CVSplits/train{i}.csv", header=None, skiprows=1)
            y_val_fold = val_data.iloc[:, 0].values.astype(int)
            X_val_fold = val_data.iloc[:, 1:].values.astype(float)

            # Training and evaluating the model
            perceptron = Perceptron(eta)
            dev_accuracies = perceptron.train(X_train, y_train, epochs_cv)
            fold_accuracies.append(dev_accuracies)

        # Average accuracy for this learning rate
        avg_accuracy = np.mean(fold_accuracies, axis=0)
        results.append((eta, avg_accuracy))

    # Choose the best learning rate
    best_eta, best_accuracy = max(results, key=lambda x: np.max(x[1]))
    return best_eta, best_accuracy

def evaluate(X, y, perceptron):
    y_pred = perceptron.predict(X)
    accuracy = np.mean(y_pred == y)
    return accuracy

# Load data
train_data = pd.read_csv("diabetes.train.csv", header=None, skiprows=1)
X_train = train_data.iloc[:, 1:].values.astype(float)
y_train = train_data.iloc[:, 0].values.astype(int)

dev_data = pd.read_csv("diabetes.dev.csv", header=None, skiprows=1)
X_dev = dev_data.iloc[:, 1:].values.astype(float)
y_dev = dev_data.iloc[:, 0].values.astype(int)

# Define hyperparameters
learning_rates = [1, 0.1, 0.01]
epochs_cv = 10

# Cross-validation on development set to find the best learning rate
best_eta, best_accuracy = cross_validation(learning_rates, epochs_cv)

#print("Best learning rate found:", best_eta)
#print("Best accuracy: ", best_accuracy)

# Train the model with the best learning rate for more epochs
perceptron = Perceptron(best_eta)
dev_accuracies = perceptron.train(X_train, y_train, epochs=20)

# Plot the learning curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(dev_accuracies) + 1), dev_accuracies, marker='o', linestyle='-')
plt.title("simple-Perceptron")
plt.xlabel("Epoch")
plt.ylabel("Dev Set Accuracy")
plt.xticks(range(1, len(dev_accuracies) + 1))
plt.grid(True)
save_folder="figures/"
plt.savefig(save_folder + 'simple_dev_accuracy_graph.png')
plt.show()
