import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# Define the hyperparameters to search over
learning_rates = [10 ** i for i in range(0, -6, -1)]
tradeoffs = [10 ** i for i in range(0, -6, -1)]
depths = [5,10]

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
print("SVM over trees\n")
# Initialize variables to store the best metrics and hyperparameters
best_f1 = -np.inf
best_params = {}

# Lists to store metrics for cross-validation
cv_precisions = []
cv_recalls = []
cv_f1_scores = []

# Perform a grid search over the hyperparameters
for lr in learning_rates:
    for C in tradeoffs:
        for d in depths:
            # Initialize the classifier with current hyperparameters
            # The learning_rate and C are not actual parameters of DecisionTreeClassifier
            # Replace the following with your actual classifier and its parameters
            classifier = DecisionTreeClassifier(max_depth=d)

            # Perform cross-validation
            for i in range(5):
                # Read the cross-validation split
                fold_data = pd.read_csv(f'/home/u1472278/ML/Mod6/hw6-data/CVSplits/training0{i}.csv')
                X_fold = fold_data.iloc[:, 1:].values
                y_fold = fold_data.iloc[:, 0].values

                # Split the fold data into training and validation sets
                X_fold_train, X_fold_val, y_fold_train, y_fold_val = train_test_split(
                    X_fold, y_fold, test_size=0.2, random_state=i)

                # Train the model
                classifier.fit(X_fold_train, y_fold_train)

                # Make predictions on the validation set
                y_val_pred = classifier.predict(X_fold_val)

                # Calculate precision, recall, and F1 score for the validation set
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_fold_val, y_val_pred, average='binary')

                # Append to lists
                cv_precisions.append(precision)
                cv_recalls.append(recall)
                cv_f1_scores.append(f1)

            # Calculate average precision, recall, and F1 score across all folds
            avg_precision = np.mean(cv_precisions)
            avg_recall = np.mean(cv_recalls)
            avg_f1 = np.mean(cv_f1_scores)

            # If the F1 score for this parameter setting is better than the best so far, update the best parameters
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_params = {'learning_rate': lr, 'C': C, 'depth': d, 'f1': best_f1,
                               'precision': avg_precision, 'recall': avg_recall}

# Train the model with the best hyperparameters on the full training set
# Replace the following with your actual classifier and its best hyperparameters
final_classifier = DecisionTreeClassifier(max_depth=best_params['depth'])
final_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_test_pred = final_classifier.predict(X_test)

# Calculate precision, recall, and F1 score for the test set
test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
    y_test, y_test_pred, average='binary')

# Output the best hyperparameters and the test set evaluation metrics
print("Best hyperparameters:", best_params)
print("Test set precision:", test_precision)
print("Test set recall:", test_recall)
print("Test set F1 score:", test_f1)
