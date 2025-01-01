import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

        # Lists to store accuracy and epochs
        accuracies = []
        epochs_list = []

        # Training loop
        for epoch in range(epochs):
            for i in range(n_samples):
                if y[i] * np.dot(X[i], self.w) <= self.margin:
                    # Compute the learning rate
                    learning_rate = (self.margin - y[i] * np.dot(X[i], self.w)) / (np.dot(X[i], X[i]) + 1)
                    # Update weights
                    self.w += learning_rate * y[i] * X[i]
                    self.total_updates += 1

            # Evaluate accuracy on dev set at the end of each epoch
            accuracy = self.evaluate(X_dev, y_dev)
            accuracies.append(accuracy)
            epochs_list.append(epoch + 1)

        return epochs_list, accuracies

    def predict(self, X):
        return np.sign(np.dot(X, self.w))

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy

def cross_validation(X_train, y_train, margins, epochs_cv):
    results = []
    for margin in margins:
        fold_accuracies = []
        for i in range(5):
            # Splitting data into train and validation sets
            val_data = pd.read_csv(f"CVSplits/train{i}.csv", header=None, skiprows=1)
            y_val_fold = val_data.iloc[:, 0].values.astype(int)
            X_val_fold = val_data.iloc[:, 1:].values.astype(float)

            # Training and evaluating the model
            perceptron = AggressivePerceptron(margin)
            epochs_list, accuracies = perceptron.train(X_train, y_train, epochs_cv)
            fold_accuracies.append(accuracies)

        # Average accuracy for this margin
        avg_accuracies = np.mean(fold_accuracies, axis=0)
        results.append((margin, epochs_list, avg_accuracies))

    # Choose the best margin
    best_margin, best_epochs, best_accuracies = max(results, key=lambda x: np.max(x[2]))
    return best_margin, best_epochs, best_accuracies

# Load data
train_data = pd.read_csv("diabetes.train.csv", header=0)
y_train = train_data.iloc[:, 0].values.astype(int)
X_train = train_data.iloc[:, 1:].values.astype(float)

# Define hyperparameters
margins = [1, 0.1, 0.01]
epochs_cv = 20

dev_data = pd.read_csv("diabetes.dev.csv", header=0)
y_dev = dev_data.iloc[:, 0].values.astype(int)
X_dev = dev_data.iloc[:, 1:].values.astype(float)

# Perform cross-validation to choose the best margin
best_margin, best_epochs, best_accuracies = cross_validation(X_train, y_train, margins, epochs_cv)

# Plot the graph
plt.figure(figsize=(10, 6))
plt.title('aggressive Development Set Accuracy per Epoch for Different Margins')
plt.xlabel('Epoch')
plt.ylabel('Development Set Accuracy')
plt.plot(best_epochs, best_accuracies, label=f"Margin: {best_margin}")
plt.legend()
plt.grid(True)



# Save the graph
save_folder="figures/"
plt.savefig(save_folder + 'aggressive_dev_accuracy_graph.png')
plt.show()