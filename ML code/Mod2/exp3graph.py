import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

class MarginPerceptron:
    def __init__(self, initial_learning_rate, margin):
        self.initial_learning_rate = initial_learning_rate
        self.margin = margin
        self.w = None
        self.b = None
        self.timestep = 0
        self.total_updates = 0  # Initialize total updates counter
        self.dev_accuracies = []  # List to store development set accuracies

    def train(self, X, y, epochs):
        n_samples, n_features = X.shape
        # Initialize weights and bias
        self.w = np.random.uniform(-0.01, 0.01, size=n_features)
        self.b = np.random.uniform(-0.01, 0.01)

        # Training loop
        for _ in range(epochs):
            epoch_accuracy = evaluate(X_dev, y_dev, self)
            self.dev_accuracies.append(epoch_accuracy)  # Record development set accuracy for this epoch
            
            for i in range(n_samples):
                self.timestep += 1
                learning_rate = self.initial_learning_rate / (1 + self.timestep)
                if y[i] * (np.dot(X[i], self.w) + self.b) < self.margin:
                    # Update weights and bias
                    self.w += learning_rate * y[i] * X[i]
                    self.b += learning_rate * y[i]
                    self.total_updates += 1  # Increment total updates counter

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

# Cross-validation function for Margin Perceptron
def cross_validation(X_train, y_train, hyperparams, epochs_cv):
    results = []
    for eta0 in hyperparams['initial_learning_rate']:
        for margin in hyperparams['margin']:
            fold_accuracies = []
            for i in range(5):
                # Splitting data into train and validation sets
                val_data = pd.read_csv(f"CVSplits/train{i}.csv", header=None, skiprows=1)
                y_val_fold = val_data.iloc[:, 0].values.astype(int)
                X_val_fold = val_data.iloc[:, 1:].values.astype(float)

                # Training and evaluating the model
                perceptron = MarginPerceptron(eta0, margin)
                perceptron.train(X_train, y_train, epochs_cv)
                accuracy = evaluate(X_val_fold, y_val_fold, perceptron)
                fold_accuracies.append(accuracy)

            # Average accuracy for this hyperparameter combination
            avg_accuracy = np.mean(fold_accuracies)
            results.append(((eta0, margin), avg_accuracy))

    # Choose the best hyperparameter combination
    best_hyperparams, best_accuracy = max(results, key=lambda x: x[1])
    return best_hyperparams, best_accuracy

# Evaluation function
def evaluate(X, y, perceptron):
    y_pred = perceptron.predict(X)
    accuracy = np.mean(y_pred == y)
    return accuracy

# Load data
train_data = pd.read_csv("diabetes.train.csv", header=None, skiprows=1)
X_train = train_data.iloc[:, 1:].values.astype(float)
y_train = train_data.iloc[:, 0].values.astype(int)

epochs_cv = 10
epochs_train = 20

dev_data = pd.read_csv("diabetes.dev.csv", header=None, skiprows=1)
X_dev = dev_data.iloc[:, 1:].values.astype(float)
y_dev = dev_data.iloc[:, 0].values.astype(int)

# Define hyperparameters
hyperparams = {
    'initial_learning_rate': [1, 0.1, 0.01],
    'margin': [1, 0.1, 0.01]
}

# Cross-validation on development set to find the best hyperparameter combination
best_hyperparams, best_accuracy = cross_validation(X_train, y_train, hyperparams, epochs_cv)

# print("Best hyperparameters found:", best_hyperparams)
# print("Best accuracy:", best_accuracy)

best_eta0, best_margin = best_hyperparams
perceptron = MarginPerceptron(best_eta0, best_margin)
perceptron.train(X_train, y_train, epochs_train)

# print("Total number of updates during training:", perceptron.total_updates)

test_data = pd.read_csv("diabetes.test.csv", header=None, skiprows=1)
X_test = test_data.iloc[:, 1:].values.astype(float)
y_test = test_data.iloc[:, 0].values.astype(int)
test_accuracy = evaluate(X_test, y_test, perceptron)
# print("Test accuracy:", test_accuracy)

dev_accuracy = evaluate(X_dev, y_dev, perceptron)
# print("Dev accuracy:", dev_accuracy)

# Plot the development set accuracy over epochs
plt.plot(range(1, epochs_train + 1), perceptron.dev_accuracies, marker='o')
plt.title('Development Set Accuracy over Epochs (Margin Perceptron)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
save_folder="figures/"
plt.savefig(save_folder + 'margin_dev_accuracy_graph.png')

plt.show()
