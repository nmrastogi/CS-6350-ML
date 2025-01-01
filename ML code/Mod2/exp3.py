import numpy as np
import pandas as pd
np.random.seed(42)
# Perceptron class
class MarginPerceptron:
    def __init__(self, initial_learning_rate, margin):
        self.initial_learning_rate = initial_learning_rate
        self.margin = margin
        self.w = None
        self.b = None
        self.timestep = 0
        self.total_updates = 0  # Initialize total updates counter

    def train(self, X, y, epochs):
        n_samples, n_features = X.shape
        # Initialize weights and bias
        self.w = np.random.uniform(-0.01, 0.01, size=n_features)
        self.b = np.random.uniform(-0.01, 0.01)

        # Training loop
        for _ in range(epochs):
            for i in range(n_samples):
                self.timestep += 1
                learning_rate = self.initial_learning_rate / (1 + self.timestep)
                if y[i] * (np.dot(X[i], self.w) + self.b) < self.margin:
                    # Update weights and bias
                    self.w += learning_rate * y[i] * X[i]
                    self.b += learning_rate * y[i]
                    self.total_updates += 1  # Increment total updates counter
    #         dev_accuracy = evaluate(X_dev, y_dev, self)
    #     #print(f"Epoch {_ + 1}/{epochs_train}, Development Accuracy: {dev_accuracy}")

    # # Check if this epoch achieves the highest development accuracy
    #     if dev_accuracy > best_dev_accuracy:
    #         best_dev_accuracy = dev_accuracy
    #         best_perceptron = MarginPerceptron(self.initial_learning_rate)
    #         best_perceptron.w = self.w.copy()
    #         best_perceptron.b = self.b

    #     return best_perceptron

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
                val_data = pd.read_csv(f"CVSplits/train{i}.csv", header=None,skiprows=1)
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
train_data = pd.read_csv("diabetes.train.csv", header=0)
y_train = train_data.iloc[:, 0].values.astype(int)
X_train = train_data.iloc[:, 1:].values.astype(float)

epochs_cv = 10
epochs_train = 20

dev_data = pd.read_csv("diabetes.dev.csv", header=0)
y_dev = dev_data.iloc[:, 0].values.astype(int)
X_dev = dev_data.iloc[:, 1:].values.astype(float)

# Define hyperparameters
hyperparams = {
    'initial_learning_rate': [1, 0.1, 0.01],
    'margin': [1, 0.1, 0.01]
}

# Cross-validation on development set to find the best hyperparameter combination
best_hyperparams, best_accuracy = cross_validation(X_train, y_train, hyperparams, epochs_cv)

print("Margin perceptron")
print("Best hyperparameters found:", best_hyperparams)
print("Best accuracy:", best_accuracy)

best_eta0, best_margin = best_hyperparams
perceptron = MarginPerceptron(best_eta0, best_margin)
perceptron.train(X_train, y_train, epochs_train)

print("Total number of updates during training:", perceptron.total_updates)

test_data = pd.read_csv("diabetes.test.csv", header=0)
y_test = test_data.iloc[:, 0].values.astype(int)
X_test = test_data.iloc[:, 1:].values.astype(float)
test_accuracy = evaluate(X_test, y_test, perceptron)
print("Test accuracy:", test_accuracy)


dev_accuracy = evaluate(X_dev, y_dev, perceptron)
print("Dev accuracy:", dev_accuracy)

#perceptron = MarginPerceptron(best_eta0)
# best_perceptron = perceptron.train(X_train, y_train, epochs_train)

# test_accuracy = evaluate(X_test, y_test, best_perceptron)
# print("Test accuracy using the best-performing perceptron:", test_accuracy)
