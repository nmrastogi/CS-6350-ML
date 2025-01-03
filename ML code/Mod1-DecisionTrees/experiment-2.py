import pandas as pd
import numpy as np
def replace_missing_stalk_root(csv_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Find the most common value in the 'stalk-root' column (excluding '?')
    most_common_value = df['stalk-root'][df['stalk-root'] != '?'].mode().iloc[0]

    # Replace '?' with the most common value in the 'stalk-root' column
    df['stalk-root'] = df['stalk-root'].replace('?', most_common_value)

    # Overwrite the existing CSV file with the modified DataFrame
    df.to_csv(csv_file, index=False)

    #print(f"Replaced missing values in 'stalk-root' column and overwritten the existing file at {csv_file}")

# Iterate over folds from 1 to 5
for fold in range(1, 6):
    csv_file = f"CVfolds_new/fold{fold}.csv"
    #print(f"Replacing missing values in 'stalk-root' column for fold {fold}:")
    replace_missing_stalk_root(csv_file)
    #print()


def calc_total_entropy(train_data, label, class_list):
    total_row = train_data.shape[0] #the total size of the dataset
    total_entr = 0
    
    for c in class_list: #for each class in the label
        total_class_count = train_data[train_data[label] == c].shape[0] #number of the class
        total_class_entr = - (total_class_count/total_row)*np.log2(total_class_count/total_row) #entropy of the class
        total_entr += total_class_entr #adding the class entropy to the total entropy of the dataset
    
    return total_entr

def calc_entropy(feature_value_data, label, class_list):
    class_count = feature_value_data.shape[0]
    entropy = 0
    
    for c in class_list:
        label_class_count = feature_value_data[feature_value_data[label] == c].shape[0] #row count of class c 
        entropy_class = 0
        if label_class_count != 0:
            probability_class = label_class_count/class_count #probability of the class
            entropy_class = - probability_class * np.log2(probability_class)  #entropy
        entropy += entropy_class
    return entropy

def calc_info_gain(feature_name, train_data, label, class_list):
    feature_value_list = train_data[feature_name].unique() #unqiue values of the feature
    total_row = train_data.shape[0]
    feature_info = 0.0
    
    for feature_value in feature_value_list:
        feature_value_data = train_data[train_data[feature_name] == feature_value] #filtering rows with that feature_value
        feature_value_count = feature_value_data.shape[0]
        feature_value_entropy = calc_entropy(feature_value_data, label, class_list) #calculcating entropy for the feature value
        feature_value_probability = feature_value_count/total_row
        feature_info += feature_value_probability * feature_value_entropy #calculating information of the feature value
        
    return calc_total_entropy(train_data, label, class_list) - feature_info #calculating information gain by subtracting

def find_most_informative_feature(train_data, label, class_list):
    feature_list = train_data.columns.drop(label) #finding the feature names in the dataset
                                            #N.B. label is not a feature, so dropping it
    max_info_gain = -1
    max_info_feature = None
    
    for feature in feature_list:  #for each feature in the dataset
        feature_info_gain = calc_info_gain(feature, train_data, label, class_list)
        if max_info_gain < feature_info_gain: #selecting feature name with highest information gain
            max_info_gain = feature_info_gain
            max_info_feature = feature
            
    return max_info_feature

def generate_sub_tree(feature_name, train_data, label, class_list):
    feature_value_count_dict = train_data[feature_name].value_counts(sort=False) #dictionary of the count of unqiue feature value
    tree = {} #sub tree or node
    
    for feature_value, count in feature_value_count_dict.iteritems():
        feature_value_data = train_data[train_data[feature_name] == feature_value] #dataset with only feature_name = feature_value
        
        assigned_to_node = False #flag for tracking feature_value is pure class or not
        for c in class_list: #for each class
            class_count = feature_value_data[feature_value_data[label] == c].shape[0] #count of class c

            if class_count == count: #count of (feature_value = count) of class (pure class)
                tree[feature_value] = c #adding node to the tree
                train_data = train_data[train_data[feature_name] != feature_value] #removing rows with feature_value
                assigned_to_node = True
        if not assigned_to_node: #not pure class
            tree[feature_value] = "?" #as feature_value is not a pure class, it should be expanded further, 
                                      #so the branch is marking with ?
            
    return tree, train_data

def make_tree(root, prev_feature_value, train_data, label, class_list):
    if train_data.shape[0] != 0: #if dataset becomes enpty after updating
        max_info_feature = find_most_informative_feature(train_data, label, class_list) #most informative feature
        tree, train_data = generate_sub_tree(max_info_feature, train_data, label, class_list) #getting tree node and updated dataset
        next_root = None
        
        if prev_feature_value != None: #add to intermediate node of the tree
            root[prev_feature_value] = dict()
            root[prev_feature_value][max_info_feature] = tree
            next_root = root[prev_feature_value][max_info_feature]
        else: #add to root of the tree
            root[max_info_feature] = tree
            next_root = root[max_info_feature]
        
        for node, branch in list(next_root.items()): #iterating the tree node
            if branch == "?": #if it is expandable
                feature_value_data = train_data[train_data[max_info_feature] == node] #using the updated dataset
                make_tree(next_root, node, feature_value_data, label, class_list) #recursive call with updated dataset

def id3(train_data_m, label, max_depth=None):
    train_data = train_data_m.copy() 
    tree = {} 
    class_list = train_data[label].unique() 
    make_tree(tree, None, train_data, label, class_list) 
    
    # Additional information retrieval
    root_feature = next(iter(tree))
    information_gain = calc_info_gain(root_feature, train_data, label, class_list)
    max_tree_depth = get_max_depth(tree)
    
    if max_depth is not None:
        tree = prune_tree(tree, max_depth)
    
    return tree, root_feature, information_gain, max_tree_depth

def get_max_depth(tree):
    if not isinstance(tree, dict):
        return 1
    else:
        depths = []
        for branch in tree.values():
            depths.append(get_max_depth(branch))
        return 1 + max(depths)

def prune_tree(tree, max_depth):
    if isinstance(tree, dict):
        if max_depth == 1:
            return "?"  # Replace subtree with a placeholder
        else:
            pruned_tree = {}
            for key, value in tree.items():
                pruned_tree[key] = prune_tree(value, max_depth - 1)
            return pruned_tree
    else:
        return tree

def predict(tree, instance):
    if not isinstance(tree, dict): #if it is leaf node
        return tree #return the value
    else:
        root_node = next(iter(tree)) #getting first key/feature name of the dictionary
        feature_value = instance[root_node] #value of the feature
        if feature_value in tree[root_node]: #checking the feature value in current tree node
            return predict(tree[root_node][feature_value], instance) #goto next feature
        else:
            return None

def evaluate(tree, test_data_m, label):
    correct_preditct = 0
    wrong_preditct = 0
    for index, row in test_data_m.iterrows(): #for each row in the dataset
        result = predict(tree, test_data_m.iloc[index]) #predict the row
        if result == test_data_m[label].iloc[index]: #predicted value and expected value is same or not
            correct_preditct += 1 #increase correct count
        else:
            wrong_preditct += 1 #increase incorrect count
    accuracy = correct_preditct / (correct_preditct + wrong_preditct) #calculating accuracy
    return accuracy

# Load the dataset
train_data_m = pd.read_csv("train.csv")

# Perform 5-fold cross-validation with fixed depth

max_depths = [1, 2, 3, 4, 5, 10, 15]  # Specify the fixed depth here
for max_depth in max_depths:
    fold_accuracies = []
    for fold in range(1, 6):
    # Split data into train and test sets for this fold
        test_data = pd.read_csv(f"CVfolds_new/fold{fold}.csv")
        train_data = pd.concat([pd.read_csv(f"CVfolds_new/fold{i}.csv") for i in range(1, 6) if i != fold])
    
    # Train the model
        tree, _, _, _ = id3(train_data, 'label', max_depth=max_depth)
    
    # Evaluate the model
        fold_accuracy = evaluate(tree, test_data, 'label')
        fold_accuracies.append(fold_accuracy)

# Calculate average accuracy
    average_accuracy = np.mean(fold_accuracies)
    std_deviation = np.std(fold_accuracies)

    print("Average Cross-Validation Accuracy with Max Depth", max_depth, ":", average_accuracy)
    print("Standard Deviation of Cross-Validation Accuracy:", std_deviation)
