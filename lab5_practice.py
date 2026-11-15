import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report,r2_score
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split

df=pd.read_csv('electricity_cost_dataset.csv')
le=LabelEncoder()
df['structure type']=le.fit_transform(df['structure type'])
mode=df['structure type'].mode().iloc[0]
df['structure type'].fillna(mode)
df.fillna(np.mean(df),inplace=True)
scale=StandardScaler()
X=df.drop(columns=['electricity cost'])
y=df['electricity cost']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

X_train=scale.fit_transform(X_train)
X_test=scale.transform(X_test)

X1=np.array(X_train).T
Y1=np.array(y_train)
W1=np.random.randn(1,X1.shape[0])
B1=np.zeros((1,1))
epochs=2000
alpha=0.01
for epoch in range(epochs):
    Y_hat=np.dot(W1,X1)+B1
    error=(Y_hat-Y1)
    loss=(1/2*X1.shape[1])*np.sum(error**2)

    dW1=(1/X1.shape[1])*np.dot(error,X1.T)
    dB1=(1/X1.shape[1])*np.sum(error)

    W1=W1-alpha*dW1
    B1=B1-alpha*dB1

y_pred=np.dot(W1,X_test.T)+B1
y_pred=y_pred.flatten()
error=y_pred-np.array(y_test)
loss=(1/2*X_test.shape[0])*np.sum(error**2)

print("The accuracy of this manual linear reggression is: ")
print(r2_score(y_test,y_pred))

model=LinearRegression()
model.fit(X_train,y_train)
prediction=model.predict(X_test)
print("The accuracy of the model implemented using the library is :")
prediction=prediction.flatten()
print(r2_score(y_test,prediction))