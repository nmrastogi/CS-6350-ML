import numpy as np
import pandas as pd

class DecisionTreeID3:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, features, target):
        data = features.copy()
        data[target.name] = target
        self.tree = self._id3(data, target.name, data.columns[:-1], depth=0)

    def _id3(self, data, target, attributes, depth):
        if len(np.unique(data[target])) == 1:
            return data[target].iloc[0]
        if len(attributes) == 0 or (self.max_depth is not None and depth >= self.max_depth):
            return data[target].mode()[0]
        else:
            best_attribute = self._select_best_attribute(data, target, attributes)
            tree = {best_attribute: {}}
            depth += 1
            for value in np.unique(data[best_attribute]):
                sub_data = data[data[best_attribute] == value].drop(columns=[best_attribute])
                subtree = self._id3(sub_data, target, sub_data.columns[:-1], depth)
                tree[best_attribute][value] = subtree
            return tree

    def _select_best_attribute(self, data, target, attributes):
        best_gain = -np.inf
        best_attribute = None
        for attribute in attributes:
            gain = self._information_gain(data, target, attribute)
            if gain > best_gain:
                best_gain = gain
                best_attribute = attribute
        return best_attribute

    def _information_gain(self, data, target, attribute):
        total_entropy = self._entropy(data[target])
        values, counts = np.unique(data[attribute], return_counts=True)
        weighted_entropy = sum((counts[i] / np.sum(counts)) * self._entropy(data[data[attribute] == values[i]][target]) for i in range(len(values)))
        return total_entropy - weighted_entropy

    def _entropy(self, data):
        probabilities = data.value_counts() / len(data)
        if probabilities.empty:
            return 0
        return -sum(probabilities * np.log2(probabilities))

    def predict(self, features):
        results = []
        for _, row in features.iterrows():
            results.append(self._predict(self.tree, row))
        return np.array(results)

    def _predict(self, tree, instance):
        if not isinstance(tree, dict):
            return tree
        attribute = next(iter(tree))
        if instance[attribute] in tree[attribute]:
            return self._predict(tree[attribute][instance[attribute]], instance)
        else:
            return np.nan  # Handle missing attribute cases

# Implementation to load data, train the model, and predict would follow this class definition


def load_data(filepath):
    data = pd.read_csv(filepath)
    features = data.iloc[:, 1:]  # Assuming first column is the label
    target = data.iloc[:, 0]    # First column is the target
    return features, target

def precision_recall_f1(predictions, actual):
    tp = np.sum((predictions == 1) & (actual == 1))
    fp = np.sum((predictions == 1) & (actual == 0))
    fn = np.sum((predictions == 0) & (actual == 1))
    
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    return precision, recall, f1

def evaluate_hyperparameters(train_files, test_files, learning_rates, Cs, depths, epochs=100, batch_size=32):
    best_f1 = -np.inf
    best_params = None
    results = {'precision': [], 'recall': [], 'f1': []}

    for lr in learning_rates:
        for C in Cs:
            for depth in depths:
                fold_precisions = []
                fold_recalls = []
                fold_f1s = []

                for train_file, test_file in zip(train_files, test_files):
                    train_features, train_target = load_data(train_file)
                    test_features, test_target = load_data(test_file)

                    tree = DecisionTreeID3(max_depth=depth)
                    for _ in range(epochs):
                        for start in range(0, len(train_features), batch_size):
                            end = min(start + batch_size, len(train_features))
                            batch_features = train_features[start:end]
                            batch_target = train_target[start:end]
                            tree.fit(batch_features, batch_target)

                    predictions = tree.predict(test_features)
                    precision, recall, f1 = precision_recall_f1(predictions, test_target)
                    fold_precisions.append(precision)
                    fold_recalls.append(recall)
                    fold_f1s.append(f1)

                avg_precision = np.mean(fold_precisions)
                avg_recall = np.mean(fold_recalls)
                avg_f1 = np.mean(fold_f1s)

                results['precision'].append(avg_precision)
                results['recall'].append(avg_recall)
                results['f1'].append(avg_f1)

                if avg_f1 > best_f1:
                    best_f1 = avg_f1
                    best_params = {'learning_rate': lr, 'C': C, 'depth': depth}

    return best_params, results


def main():
    fold_paths = ["CVSplits/training00.csv", "CVSplits/training01.csv", "CVSplits/training02.csv", "CVSplits/training03.csv", "CVSplits/training04.csv"]
    learning_rates = [10 ** i for i in range(0, -6, -1)]
    Cs = [10 ** i for i in range(0, -6, -1)]
    depths = [5, 10]

    best_params, cv_results = evaluate_hyperparameters(fold_paths[:-1], [fold_paths[-1]], learning_rates, Cs, depths,epochs=100, batch_size=32)
    print("Best Hyperparameters found:", best_params)
    print("Cross-Validation Average Metrics: Precision: {:.2f}, Recall: {:.2f}, F1 Score: {:.2f}".format(
        np.mean(cv_results['precision']), np.mean(cv_results['recall']), np.mean(cv_results['f1'])))

    # Train the final model using all training data and the best hyperparameters
    X_train, y_train = load_data('train.csv')
    X_test, y_test = load_data('test.csv')

    tree = DecisionTreeID3(max_depth=best_params['depth'])
    tree.fit(X_train, y_train)
    predictions = tree.predict(X_test)

    test_precision, test_recall, test_f1 = precision_recall_f1(predictions, y_test)
    print("Test Data Metrics: Precision: {:.4f}, Recall: {:.4f}, F1 Score: {:.4f}".format(test_precision, test_recall, test_f1))

if __name__ == "__main__":
    main()