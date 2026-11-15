import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

df=pd.read_csv("mushrooms.csv")
df.replace('?',np.nan,inplace=True)

mode_e=df[df['class']=='e'].mode().iloc[0]
mode_p=df[df['class']=='p'].mode().iloc[0]

df.update(df[df['class'] == 'e'].fillna(mode_e))
df.update(df[df['class'] == 'p'].fillna(mode_p))

train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

def entropy(feature):
    ent=0
    tlen=len(feature)
    for c in feature.unique():
        prob=(feature==c).sum()/tlen
        if prob>0:
            ent-=prob*np.log2(prob)
    return ent

def info_gain(dataset,feature,target):
    tlen=len(dataset)
    ent=entropy(dataset[target])
    weighted_ent=0
    for c in dataset[feature].unique():
        subset=dataset[dataset[feature]==c]
        weight=len(subset)/tlen
        temp=entropy(subset[target])
        weighted_ent+=weight*temp
    inf=ent-weighted_ent
    return inf

def gain_ratio(dataset,feature,target):
    tlen=len(dataset)
    inf=info_gain(dataset,feature,target)
    split_info=0
    for c in dataset[feature].unique():
        subset=dataset[dataset[feature]==c]
        weight=len(subset)/tlen
        if weight>0:
            split_info-=weight*(np.log2(weight))
    if split_info==0:
        return 0
    gain_r=inf/(split_info)
    return gain_r

def calc_gini(target):
    gin=1
    for val in target.unique():
        prob=(target==val).sum()/len(target)
        if prob>0:
            gin-=prob*prob
    return gin

def gini_index(dataset,feature,target):
    base_gini=calc_gini(dataset[target])
    weighted_gini=0
    for c in dataset[feature].unique():
        subset=dataset[dataset[feature]==c]
        weighted_gini+=(len(subset)/len(dataset))*calc_gini(subset[target])
    gini=base_gini-weighted_gini
    return gini

def calc_err(target):
    err=1
    maxy=0
    for val in target.unique():
        prob=(target==val).sum()/len(target)
        if prob>maxy:
            maxy=prob
    err-=maxy
    return err

def misclassification_error(dataset,feature,target):
    base_error=calc_err(dataset[target])
    weight=0
    for c in dataset[feature].unique():
        subset=dataset[dataset[feature]==c]
        prob=len(subset)/len(dataset)
        if prob>0:
            weight+=prob*calc_err(subset[target])
    error=base_error-weight
    return error

def build_id3(dataset, feature_list, target, criterion):
    if len(dataset[target].unique()) == 1:
        return dataset[target].iloc[0]
        
    if len(feature_list) == 0:
        return dataset[target].mode()[0]
        
    scores = {}
    for feature in feature_list:
        if criterion == "info_gain":
            scores[feature] = info_gain(dataset, feature, target)
        elif criterion == "gain_ratio":
            scores[feature] = gain_ratio(dataset, feature, target)
        elif criterion == "gini":
            scores[feature] = gini_index(dataset, feature, target)
        elif criterion == "error":
            scores[feature] = misclassification_error(dataset, feature, target)
            
    best_feature = max(scores, key=scores.get)
    
    tree = {best_feature: {}}
    
    remaining_features = [f for f in feature_list if f != best_feature]
    
    for val in dataset[best_feature].unique():
        subset = dataset[dataset[best_feature] == val]
        
        if len(subset) == 0:
            tree[best_feature][val] = dataset[target].mode()[0]
        else:
            tree[best_feature][val] = build_id3(subset, remaining_features, target, criterion)
            
    return tree

def predict_row(tree,row):
    if type(tree)!=dict:
        return tree
    root=list(tree.keys())[0]
    if row[root] not in tree[root]:
        return 'p'
    return predict_row(tree[root][row[root]],row)

def print_rules_dfs(tree, rule=""):
    if type(tree)!=dict:
        print(f"IF {rule[:-5]} THEN class = '{tree}'")
        return
    root=list(tree.keys())[0]
    for val,subtree in tree[root].items():
        print_rules_dfs(subtree, rule+f"{root} == '{val}' AND ")

all_features=list(train_set.columns)
all_features.remove('class')

criteria=["info_gain","gain_ratio","gini","error"]

for crit in criteria:
    print("\nCRITERION:",crit.upper())
    print("Root Node Feature Scores:")
    
    for feature in all_features:
        if crit=="info_gain":
            score=info_gain(train_set,feature,'class')
        elif crit=="gain_ratio":
            score=gain_ratio(train_set,feature,'class')
        elif crit=="gini":
            score=gini_index(train_set,feature,'class')
        elif crit=="error":
            score=misclassification_error(train_set,feature,'class')
        print(feature,"=",round(score,5))
        
    current_tree=build_id3(train_set,all_features,'class',criterion=crit)
    root_attr=list(current_tree.keys())[0]
    rule=""
    print_rules_dfs(current_tree,"")
    print("Winning Root Attribute:",root_attr)
    
    corr=0
    actual=[]
    predicted=[]
    for _,row in test_set.iterrows():
        pred=predict_row(current_tree,row)
        act=row['class']
    
        actual.append(act)
        predicted.append(pred)
    
        if pred==act:
            corr+=1
            
    acc=(corr/len(test_set))*100
    print("Tree Accuracy =",round(acc,2),"%")
    cm=confusion_matrix(actual,predicted,labels=['e','p'])
    print("Confusion Matrix:")
    print(cm)