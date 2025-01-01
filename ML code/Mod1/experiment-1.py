import numpy as np
import pandas as pd
train_data_m=pd.read_csv("train.csv")

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

def id3(train_data_m, label):
    train_data = train_data_m.copy() 
    tree = {} 
    class_list = train_data[label].unique() 
    make_tree(tree, None, train_data, label, class_list) 
    
    # Additional information retrieval
    root_feature = next(iter(tree))
    information_gain = calc_info_gain(root_feature, train_data, label, class_list)
    max_depth = get_max_depth(tree)
    
    return tree, root_feature, information_gain, max_depth
def get_max_depth(tree):
    if not isinstance(tree, dict):
        return 0  # Return 0 for leaf nodes (including labels)
    else:
        depths = []
        for branch in tree.values():
            depths.append(get_max_depth(branch))
        if depths:  # Check if depths list is not empty
            return 1 + max(depths)
        else:
            return 0  # If the tree is a leaf node (including labels), return 0

tree, root_feature, information_gain, max_depth = id3(train_data_m, 'label')
print("(a) The root feature selected by the algorithm:", root_feature)
print("(b) Information gain for the root feature:", information_gain)
print("(c) Maximum depth of the tree:", max_depth)

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
    correct_predict = 0
    wrong_predict = 0
    total_instances = len(test_data_m)
    
    for index, row in test_data_m.iterrows():
        result = predict(tree, test_data_m.iloc[index])
        if result == test_data_m[label].iloc[index]:
            correct_predict += 1
        else:
            wrong_predict += 1
    
    accuracy = correct_predict / total_instances
    print("Accuracy on Testing Set:", accuracy)
    return accuracy


# test_data_m = pd.read_csv("test.csv") #importing test dataset into dataframe

# accuracy = evaluate(tree, test_data_m, 'label') #evaluating the test dataset
# accuracy_train=evaluate(tree, train_data_m, 'label')
# print("Accuracy on Training Set:", accuracy_train)
test_data_m = pd.read_csv("test.csv")  # importing test dataset into dataframe

accuracy_test = evaluate(tree, test_data_m, 'label')  # evaluating the test dataset
accuracy_train = evaluate(tree, train_data_m, 'label')  # evaluating the training dataset

print("Accuracy on Training Set:", accuracy_train)
print("Accuracy on Testing Set:", accuracy_test)




def most_common_label(train_data, label):
    most_common = train_data[label].mode()[0]
    print("Most Common Label:", most_common)
    return most_common

# Find the most common label in the training data
most_common = most_common_label(train_data_m, 'label')

# Define a function to predict the most common label for all instances
def predict_most_common(data, label, most_common):
    return [most_common] * len(data)

# Calculate training accuracy
train_predictions = predict_most_common(train_data_m, 'label', most_common)
train_accuracy = (train_predictions == train_data_m['label']).mean()

# Calculate test accuracy
test_predictions = predict_most_common(test_data_m, 'label', most_common)
test_accuracy = (test_predictions == test_data_m['label']).mean()

print("Training Accuracy (Always Predicting Most Common Label):", train_accuracy)
print("Test Accuracy (Always Predicting Most Common Label):", test_accuracy)
total_entropy = calc_total_entropy(train_data_m, 'label', train_data_m['label'].unique())
print("The total entropy of the data is:", total_entropy)


