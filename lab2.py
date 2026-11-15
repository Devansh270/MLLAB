import pandas as pd
import math
from sklearn.model_selection import train_test_split 

mushroom_df = pd.read_csv("mushrooms.csv")
target_col = "class"

train_set, test_set = train_test_split(mushroom_df, test_size=0.2, random_state=42)

def calc_entropy(column_data):
    entropy_val = 0
    for count in column_data.value_counts():
        prob = count / len(column_data)
        entropy_val -= prob * math.log2(prob)
    return entropy_val

def calc_gain(dataset, feature_name):
    total_entropy = calc_entropy(dataset[target_col])
    for unique_val in dataset[feature_name].unique():
        subset = dataset[dataset[feature_name] == unique_val]
        total_entropy -= len(subset)/len(dataset) * calc_entropy(subset[target_col])
    return total_entropy

def build_id3(dataset, feature_list):
    if len(dataset[target_col].unique()) == 1:
        return dataset[target_col].iloc[0]

    if not feature_list:
        return dataset[target_col].mode()[0]

    print("\nInformation Gain")
    gain_scores = {}

    for feature in feature_list:
        gain_scores[feature] = calc_gain(dataset, feature)
        print(feature, "=", round(gain_scores[feature], 5))

    best_attr = max(gain_scores, key=gain_scores.get)
    print("Best Attribute:", best_attr)

    my_tree = {best_attr: {}}

    for val in dataset[best_attr].unique():
        subset = dataset[dataset[best_attr] == val]
        my_tree[best_attr][val] = build_id3(subset, [f for f in feature_list if f != best_attr])

    return my_tree

all_features = list(mushroom_df.columns)
all_features.remove(target_col)

final_tree = build_id3(train_set, all_features)

print("\nDecision Tree Built Successfully.")

def predict_row(tree_model, row_data):
    if type(tree_model) != dict:
        return tree_model
    
    root_node = list(tree_model.keys())[0]
    
    if row_data[root_node] not in tree_model[root_node]:
        return 'p' 
        
    return predict_row(tree_model[root_node][row_data[root_node]], row_data)

correct_guesses = 0
for _, row_data in test_set.iterrows():
    if predict_row(final_tree, row_data) == row_data[target_col]:
        correct_guesses += 1

print("\nCustom ID3 Accuracy =", round(correct_guesses/len(test_set)*100, 2), "%")

print("\nRules Generated.")

def print_rules(tree_model, rule_str=""):
    if type(tree_model) != dict:
        print(rule_str + " -> " + tree_model)
        return

    root_node = list(tree_model.keys())[0]

    for val in tree_model[root_node]:
        print_rules(tree_model[root_node][val], rule_str + root_node + "=" + str(val) + " ")

print_rules(final_tree)

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

encoded_df = pd.read_csv("mushrooms.csv")
for col in encoded_df.columns:
    encoded_df[col] = LabelEncoder().fit_transform(encoded_df[col])

X_features = encoded_df.drop("class", axis=1)
y_labels = encoded_df["class"]

X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=42)

print("Sklearn Accuracy:",
      DecisionTreeClassifier(criterion="entropy").fit(X_train, y_train).score(X_test, y_test)*100)