import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(z):
    # Clip input to sigmoid to prevent overflow
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def compute_gradient(X, y, w, C):
    m = X.shape[0]
    y_pred = sigmoid(np.dot(X, w)) * 2 - 1  # Transform output to -1 and +1
    gradient = np.dot(X.T, (y_pred - y)) / m + C * w
    return gradient

def compute_loss(y_true, y_pred):
    # Binary cross-entropy loss
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

def logistic_regression_SGD(X_train, y_train, learning_rate, epochs, C, batch_size=32):
    w = np.zeros(X_train.shape[1])
    num_samples = X_train.shape[0]
    losses = []  # To store loss at each epoch
    
    for epoch in range(epochs):
        # Shuffle dataset at the beginning of each epoch
        indices = np.random.permutation(num_samples)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        epoch_loss = 0
        
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            X_batch = X_train_shuffled[start:end]
            y_batch = y_train_shuffled[start:end]
            
            gradient = compute_gradient(X_batch, y_batch, w, C)
            w -= learning_rate * gradient
            
            # Compute loss for this batch and add it to the epoch loss
            y_pred_batch = sigmoid(np.dot(X_batch, w))
            batch_loss = compute_loss(y_batch, y_pred_batch)
            epoch_loss += batch_loss
        
        # Average epoch loss and add it to the losses list
        epoch_loss /= (num_samples / batch_size)
        losses.append(epoch_loss)

        learning_rate /= (1 + epoch)  # Decaying learning rate

    return w, losses

def predict(X, weights):
    z = np.dot(X, weights)
    return np.where(sigmoid(z) >= 0.5, 1, -1)  # Returns 1 or -1

def calculate_metrics(y_true, y_pred):
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    false_positives = np.sum((y_pred == 1) & (y_true == -1))
    false_negatives = np.sum((y_pred == -1) & (y_true == 1))

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1

def cross_validate(fold_paths, learning_rates, Cs, epochs, batch_size):
    best_params = {}
    best_f1 = -np.inf
    average_scores = {'precision': [], 'recall': [], 'f1': []}

    for gamma in learning_rates:
        for C in Cs:
            all_f1_scores = []
            all_precision_scores = []
            all_recall_scores = []

            for fold in fold_paths:
                fold_data = pd.read_csv(fold)
                X_train = fold_data.iloc[:, 1:].values
                y_train = fold_data.iloc[:, 0].values

                validation_folds = [f for f in fold_paths if f != fold]
                for valid_fold in validation_folds:
                    valid_data = pd.read_csv(valid_fold)
                    X_valid = valid_data.iloc[:, 1:].values
                    y_valid = valid_data.iloc[:, 0].values
                        
                    weights, _ = logistic_regression_SGD(X_train, y_train, gamma, epochs, C, batch_size)
                    y_pred = predict(X_valid, weights)
                    
                    precision, recall, f1 = calculate_metrics(y_valid, y_pred)
                    
                    all_f1_scores.append(f1)
                    all_precision_scores.append(precision)
                    all_recall_scores.append(recall)

            avg_f1 = np.mean(all_f1_scores)
            avg_precision = np.mean(all_precision_scores)
            avg_recall = np.mean(all_recall_scores)

            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_params = {'gamma': gamma, 'C': C}
                average_scores = {'precision': avg_precision, 'recall': avg_recall, 'f1': avg_f1}

    return best_params, average_scores

# Paths to the CSV files for cross-validation
fold_paths = ['/home/u1472278/ML/Mod6/hw6-data/CVSplits/training00.csv', '/home/u1472278/ML/Mod6/hw6-data/CVSplits/training01.csv', '/home/u1472278/ML/Mod6/hw6-data/CVSplits/training02.csv', '/home/u1472278/ML/Mod6/hw6-data/CVSplits/training03.csv', '/home/u1472278/ML/Mod6/hw6-data/CVSplits/training04.csv']
#CVSplits/training00.csv
learning_rates = [10**i for i in range(0, -5, -1)]
Cs = [10**i for i in range(1, -5, -1)]
epochs = 100
batch_size = 32

# Load full dataset
full_train_data = pd.read_csv('/home/u1472278/ML/Mod6/hw6-data/train.csv')
test_data = pd.read_csv('/home/u1472278/ML/Mod6/hw6-data/test.csv')

X_train_full = full_train_data.iloc[:, 1:].values
y_train_full = full_train_data.iloc[:, 0].values
X_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values

best_hyperparameters, average_scores = cross_validate(fold_paths, learning_rates, Cs, epochs, batch_size)
print("SVM\n")
print(f"Best Hyperparameters: {best_hyperparameters}")
print(f"Average Precision across CV: {average_scores['precision']:.4f}")
print(f"Average Recall across CV: {average_scores['recall']:.4f}")
print(f"Average F1 Score across CV: {average_scores['f1']:.4f}")

final_weights, losses = logistic_regression_SGD(X_train_full, y_train_full, best_hyperparameters['gamma'], epochs, best_hyperparameters['C'], batch_size)
y_pred_test = predict(X_test, final_weights)
test_precision, test_recall, test_f1 = calculate_metrics(y_test, y_pred_test)

print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")

plt.plot(range(1, epochs + 1), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
#plt.show('')
plt.savefig("/home/u1472278/ML/Mod6/hw6-data/figure/svm.png")