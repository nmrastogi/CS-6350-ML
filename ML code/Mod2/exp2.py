import numpy as np
import pandas as pd
np.random.seed(42)
# Perceptron class
class Perceptron:
    def __init__(self, initial_learning_rate):
        self.initial_learning_rate = initial_learning_rate
        self.w = None
        self.b = None
        self.timestep = 0
        self.total_updates = 0
        
        

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
                self.timestep += 1
                learning_rate = self.initial_learning_rate / (1 + self.timestep)
                if y[i] * (np.dot(X[i], self.w) + self.b) <= 0:
                    # Update weights and bias
                    self.w += learning_rate * y[i] * X[i]
                    self.b += learning_rate * y[i]
                    self.total_updates += 1 

        dev_accuracy = evaluate(X_dev, y_dev, self)
        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            best_perceptron = Perceptron(self.initial_learning_rate)
            best_perceptron.w = self.w.copy()
            best_perceptron.b = self.b

        return best_perceptron

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

# Cross-validation function
def cross_validation(X_train, y_train, initial_learning_rates, epochs_cv):
    results = []
    for eta0 in initial_learning_rates:
        fold_accuracies = []
        for i in range(5):
            # Splitting data into train and validation sets
            val_data = pd.read_csv(f"CVSplits/train{i}.csv", header=None,skiprows=1)
            y_val_fold = val_data.iloc[:, 0].values.astype(int)
            X_val_fold = val_data.iloc[:, 1:].values.astype(float)

            # Training and evaluating the model
            perceptron = Perceptron(eta0)
            perceptron.train(X_train, y_train, epochs_cv)
            accuracy = evaluate(X_val_fold, y_val_fold, perceptron)
            fold_accuracies.append(accuracy)

        # Average accuracy for this learning rate
        avg_accuracy = np.mean(fold_accuracies)
        results.append((eta0, avg_accuracy))

    # Choose the best learning rate
    best_eta0, best_accuracy = max(results, key=lambda x: x[1])
    return best_eta0, best_accuracy

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

# Hyperparameters
initial_learning_rates = [1, 0.1, 0.01]

# Cross-validation on development set to find the best initial learning rate
best_eta0, best_accuracy = cross_validation(X_train, y_train, initial_learning_rates, epochs_cv)

# Train the model with the best initial learning rate for more epochs
perceptron = Perceptron(best_eta0)
perceptron.train(X_train, y_train, epochs_train)
print("decaying perceptron")
print("Best initial learning rate found:", best_eta0)
print("Best accuracy:", best_accuracy)
print("Total number of updates during training:", perceptron.total_updates)
# Evaluate on test set
test_data = pd.read_csv("diabetes.test.csv", header=0)
y_test = test_data.iloc[:, 0].values.astype(int)
X_test = test_data.iloc[:, 1:].values.astype(float)
test_accuracy = evaluate(X_test, y_test, perceptron)
#print("Test accuracy:", test_accuracy)


dev_accuracy = evaluate(X_dev, y_dev, perceptron)
print("Dev accuracy:", dev_accuracy)

perceptron = Perceptron(best_eta0)
best_perceptron = perceptron.train(X_train, y_train, epochs_train)

test_accuracy = evaluate(X_test, y_test, best_perceptron)
print("Test accuracy using the best-performing perceptron:", test_accuracy)