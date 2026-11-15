import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import classification_report,confusion_matrix

df=pd.read_csv('mushrooms.csv')
df.replace('?',np.nan,inplace=True)
mode=df.mode().iloc[0]
df.fillna(mode,inplace=True)
df.drop_duplicates(inplace=True)
train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

p_yes=len(train_set[train_set['class']=='e'])/len(train_set)
p_no=len(train_set[train_set['class']=='p'])/len(train_set)

def given_probs(Dataframe,target):
    given={}
    subset=Dataframe[Dataframe['class']==target]
    tlen=len(subset)
    for col in subset.columns:
        given[col]={}
        for val in subset[col].unique():
            given[col][val]=len(subset[subset[col]==val])/tlen
    return given

given_y=given_probs(train_set,'e')
given_n=given_probs(train_set,'p')

def prediction(row):
    score_y=np.log(p_yes)
    score_n=np.log(p_no)
    for col in row.index:
        val=row[col]
        ans=given_y[col].get(val,0)
        if ans>0:
            score_y+=np.log(ans)
        else:
            score_y+=np.log(1e-10)
        ans=given_n[col].get(val,0)
        if ans>0:
            score_n+=np.log(ans)
        else:
            score_n+=np.log(1e-10)
    if score_y>score_n:
        return 'e'
    else:
        return 'p'

def evaluation(test):
    pred=[]
    acc=0
    for _,row in test.iterrows():
        pred.append(prediction(row))
        if row['class']==pred[-1]:
            acc+=1
    acc=acc/len(test)
    return acc,pred

acc,y_pred=evaluation(test_set)
print("report for manual bayes")
print(classification_report(test_set['class'],y_pred,labels=['e','p']))

df_encoded=df.copy()
for col in df_encoded.columns:
    le=LabelEncoder()
    df_encoded[col]=le.fit_transform(df[col])

X=df_encoded.drop(columns=['class'])
y=df_encoded['class']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=CategoricalNB()
model.fit(X_train,y_train)
preds=model.predict(X_test)
sk_acc=np.mean(preds==y_test)

print("report for the library implemented naivebayes")
print(classification_report(y_test,preds,labels=[0,1]))