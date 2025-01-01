import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

# Define the hyperparameters to search over
learning_rates = [10 ** i for i in range(0, -6, -1)]
tradeoffs = [10 ** i for i in range(0, -6, -1)]
depths = [5, 10]

# Paths to the CSV files
train_csv_path = '/home/u1472278/ML/Mod6/hw6-data/train.csv'  # replace with your train CSV file path
test_csv_path = '/home/u1472278/ML/Mod6/hw6-data/test.csv'  # replace with your test CSV file path

# Read the train and test data
train_data = pd.read_csv(train_csv_path)
test_data = pd.read_csv(test_csv_path)

# Separate features and labels
X_train = train_data.iloc[:, 1:].values
y_train = train_data.iloc[:, 0].values
X_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values

# Initialize classifier with SGD for logistic regression
classifier = SGDClassifier(loss='log', learning_rate='constant', eta0=0.01, max_iter=1, tol=None, warm_start=True)

# List to store the loss at each epoch
losses = []

# Number of epochs
epochs = 100

# Training loop
for epoch in range(epochs):
    classifier.partial_fit(X_train, y_train, classes=np.unique(y_train))
    probabilities = classifier.predict_proba(X_train)
    loss = log_loss(y_train, probabilities)
    losses.append(loss)
    # print(f"Epoch {epoch + 1}, Loss: {loss}")

# Plot the loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), losses, label='Training Loss')
plt.title('Log Loss at Each Epoch')
plt.xlabel('Epoch')
plt.ylabel('Log Loss')
plt.legend()
plt.savefig("/home/u1472278/ML/Mod6/hw6-data/figure/svmovertrees.png")
print("done")
plt.show()
