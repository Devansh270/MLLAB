import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from pprint import pprint
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import confusion_matrix

df=pd.read_csv("mushrooms.csv")
df.replace('?',np.nan,inplace=True)

# mode_e=df[df['class']=='e'].mode().iloc[0]
# mode_p=df[df['class']=='p'].mode().iloc[0]
mode=df.mode().iloc[0]

# df.update(df[df['class'] == 'e'].fillna(mode_e))
# df.update(df[df['class'] == 'p'].fillna(mode_p))

df.fillna(mode)

df.drop_duplicates(inplace=True)
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

p_yes=len(train_set[train_set['class']=='e'])/len(train_set['class'])
p_no=len(train_set[train_set['class']=='p'])/len(train_set['class'])

features=train_set.columns
features=features.drop('class')

def given(dataset,target_class):
    given_y={}
    subset=dataset[dataset['class']==target_class]
    tlen=len(subset)
    for col in dataset.columns:
        given_y[col]={}
        for val in dataset[col].unique():
            count=(subset[col]==val).sum()
            given_y[col][val]=count/tlen
    return given_y


given_y = given(train_set,'e')
given_n = given(train_set,'p')


def predict_single(row):
    score_y = np.log(p_yes)
    score_n = np.log(p_no)
    
    for col in features:
        val = row[col]
        
        prob_y = given_y[col].get(val, 0)
        prob_n = given_n[col].get(val, 0)
        
        if prob_y > 0:
            score_y += np.log(prob_y)
        else:
            score_y += np.log(1e-10)
            
        if prob_n > 0:
            score_n += np.log(prob_n)
        else:
            score_n += np.log(1e-10)
        print(f"edible: {np.exp(score_y)} poisonous: {np.exp(score_n)}\n")
    return 'e' if score_y > score_n else 'p'

def evaluate(test_data):
    predictions = []
    accuracy=0
    for _,row in test_data.iterrows():
        pred = predict_single(row)
        predictions.append(pred)
        if pred==row['class']:
            accuracy+=1
    accuracy=(accuracy/len(test_data['class']))
    return accuracy, predictions

accuracy, y_pred = evaluate(test_set)
print("Precomputed Conditional Probablity of all columns given edible:\n")
pprint(given_y)
print("\nPrecomputed Conditional Probablity of all columns given poisonous:\n")
pprint(given_n)

df_encoded = df.copy()
for col in df_encoded.columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])

X = df_encoded[features]
y = df_encoded['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sk_model = CategoricalNB()
sk_model.fit(X_train, y_train)
sk_predictions = sk_model.predict(X_test)
sk_accuracy = np.mean(sk_predictions == y_test)

print(f"Test Accuracy(manual): {accuracy * 100:.2f}%")
print(f"\nScikit-Learn Model Accuracy: {sk_accuracy * 100:.2f}%")

print("\nManual Model Confusion Matrix:")
print(confusion_matrix(test_set['class'], y_pred, labels=['e', 'p']))

print("\nScikit-Learn Confusion Matrix:")
print(confusion_matrix(y_test, sk_predictions, labels=[0, 1]))