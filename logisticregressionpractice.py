import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
df=pd.read_csv('mushrooms.csv')
df.replace('?',np.nan,inplace=True)
mode=df.mode().iloc[0]
df.fillna(mode,inplace=True)
df.drop_duplicates(inplace=True)
X=pd.get_dummies(df.drop(columns=['class']),drop_first=True,dtype=int)
y=df['class'].astype('category').cat.codes
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X1=np.array(X_train).T
Y1=np.array(y_train)
W1=np.random.randn(1,X1.shape[0])
B1=np.zeros((1,1))
epochs=5000
alpha=0.5
for epoch in range(epochs):
    Z=np.dot(W1,X1)+B1
    y_hat=1/(1+np.exp(-Z))
    loss=(-1/X1.shape[1])*(Y1*np.log(y_hat)+(1-Y1)*np.log(1-y_hat))
    dW1=(1/X1.shape[1])*np.dot((y_hat-Y1),X1.T)
    dB1=(1/X1.shape[1])*np.sum(y_hat-Y1)
    W1-=alpha*dW1
    B1-=alpha*dB1

Z1=np.dot(W1,X_test.T)+B1
y_hat=1/(1+np.exp(-Z1))
y_hat=(y_hat>=0.5).astype(int).flatten()
print("the report for the manual implementation of logistic regression")
print(classification_report(y_test,y_hat))

model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("the report for library implementation of logistic regression")
print(classification_report(y_test,y_pred))