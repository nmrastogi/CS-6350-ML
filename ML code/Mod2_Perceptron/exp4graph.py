import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

class AveragedPerceptron:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.w = None
        self.b = None
        self.a = None  # Averaged weight vector
        self.ba = None  # Averaged bias term

    def train(self, X, y, epochs):
        n_samples, n_features = X.shape
        # Initialize weights and bias
        self.w = np.random.uniform(-0.01, 0.01, size=n_features)
        self.b = np.random.uniform(-0.01, 0.01)
        self.a = np.zeros(n_features)
        self.ba = 0

        # Initialize update counters
        total_updates = 0
        total_updates_per_epoch = np.zeros(epochs)
        accuracies_per_epoch = []

        # Training loop
        for epoch in range(epochs):
            updates = 0
            for i in range(n_samples):
                if y[i] * (np.dot(X[i], self.w) + self.b) <= 0:
                    # Update weights and bias
                    self.w += self.learning_rate * y[i] * X[i]
                    self.b += self.learning_rate * y[i]
                    self.a += self.w
                    self.ba += self.b
                    updates += 1
                    total_updates += 1

            total_updates_per_epoch[epoch] = updates

            # Evaluate accuracy on dev set at the end of each epoch
            accuracy = self.evaluate_dev_set(X_dev, y_dev)
            accuracies_per_epoch.append(accuracy)

        # Averaging weights and bias
        self.a /= total_updates
        self.ba /= total_updates

        return total_updates_per_epoch, accuracies_per_epoch

    def predict(self, X):
        return np.sign(np.dot(X, self.a) + self.ba)

    def evaluate_dev_set(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy

def cross_validation(X_train, y_train, learning_rates, epochs_cv):
    results = []
    for eta in learning_rates:
        fold_updates = []
        fold_accuracies = []
        for i in range(5):
            # Splitting data into train and validation sets
            val_data = pd.read_csv(f"CVSplits/train{i}.csv", header=None, skiprows=1)
            y_val_fold = val_data.iloc[:, 0].values.astype(int)
            X_val_fold = val_data.iloc[:, 1:].values.astype(float)

            # Training and evaluating the model
            perceptron = AveragedPerceptron(eta)
            total_updates_per_epoch, _ = perceptron.train(X_train, y_train, epochs_cv)
            fold_updates.append(total_updates_per_epoch)

            # Evaluate accuracy on validation set
            accuracy = perceptron.evaluate_dev_set(X_val_fold, y_val_fold)
            fold_accuracies.append(accuracy)

        # Average updates for this learning rate
        avg_updates = np.mean(fold_updates, axis=0)
        avg_accuracy = np.mean(fold_accuracies)
        results.append((eta, avg_updates, avg_accuracy))

    # Choose the best learning rate
    best_eta, _, best_accuracy = max(results, key=lambda x: x[2])
    return best_eta, best_accuracy

# Load data
train_data = pd.read_csv("diabetes.train.csv", header=0)
y_train = train_data.iloc[:, 0].values.astype(int)
X_train = train_data.iloc[:, 1:].values.astype(float)

dev_data = pd.read_csv("diabetes.dev.csv", header=0)
y_dev = dev_data.iloc[:, 0].values.astype(int)
X_dev = dev_data.iloc[:, 1:].values.astype(float)

# Define hyperparameters
learning_rates = [1, 0.1, 0.01]
epochs_cv = 10

# Perform cross-validation to choose the best learning rate
best_learning_rate, best_accuracy = cross_validation(X_train, y_train, learning_rates, epochs_cv)
# print("Best learning rate found:", best_learning_rate)
# print("Best accuracy found:", best_accuracy)

# Train the model with the best learning rate for 20 epochs
epochs_train = 20
perceptron = AveragedPerceptron(best_learning_rate)
total_updates_per_epoch, accuracies_per_epoch = perceptron.train(X_train, y_train, epochs_train)

# print("Total number of updates per epoch:", total_updates_per_epoch)

# Plotting the graph
plt.plot(range(1, epochs_train + 1), accuracies_per_epoch, marker='o')
plt.title('average Development Set Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Development Set Accuracy')
plt.grid(True)
save_folder="figures/"
plt.savefig(save_folder + 'average_dev_accuracy_graph.png')
plt.show()
