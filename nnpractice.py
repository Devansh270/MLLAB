import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

df=pd.read_csv('mushrooms.csv')
df.replace('?',np.nan,inplace=True)
mode=df.mode().iloc[0]
df.fillna(mode,inplace=True)
df.drop_duplicates(inplace=True)

X=pd.get_dummies(df.drop(columns=['class']),drop_first=True,dtype=int)
y=df['class'].astype('category').cat.codes

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

X1=np.array(X_train).T
Yt=np.array(y_train)
W1=np.random.rand(16,X1.shape[0])
B1=np.zeros((16,1))
W2=np.random.rand(8,W1.shape[0])
B2=np.zeros((8,1))
W3=np.random.rand(1,W2.shape[0])
B3=np.zeros((1,1))

epochs=5000
alpha=0.01
for epoch in range(epochs):
    Z1=np.dot(W1,X1)+B1
    Y1=np.maximum(0,Z1)
    Z2=np.dot(W2,Y1)+B2
    Y2=np.maximum(0,Z2)
    Z3=np.dot(W3,Y2)+B3
    Y3=1/(1+np.exp(-Z3))
    Y3=np.clip(Y3,1e-10,1-1e-10)
    loss=(-1/X1.shape[1])*np.sum(Yt*np.log(Y3)+(1-Yt)*np.log(1-Y3))

    dW3=(1/X1.shape[1])*np.dot(Y3-Yt,Y2.T)
    dB3=(1/X1.shape[1])*np.sum(Y3-Yt)
    dW2=(1/X1.shape[1])*(np.dot(W3.T,np.dot(Y3-Yt,Y1.T)))
    dB2=(1/X1.shape[1])*np.sum(np.dot(W3.T,Y3-Yt))
    dW1=(1/X1.shape[1])*(np.dot(W2.T,np.dot(W3.T,np.dot(Y3-Yt,X1.T))))
    dB1=(1/X1.shape[1])*np.sum(np.dot(W2.T,np.dot(W3.T,Y3-Yt)))

    W1-=alpha*dW1
    B1-=alpha*dB1
    W2-=alpha*dW2
    B2-=alpha*dB2
    W3-=alpha*dW3
    B3-=alpha*dB3

Z1=np.dot(W1,X_test.T)+B1
Y1=np.maximum(0,Z1)
Z2=np.dot(W2,Y1)+B2
Y2=np.maximum(0,Z2)
Z3=np.dot(W3,Y2)+B3
Y3=1/(1+np.exp(-Z3))
ypred=(Y3>=0.5).astype(int).flatten()
print("report for manual implementation of neural network")
print(classification_report(y_test,ypred))

model=MLPClassifier(hidden_layer_sizes=(16,8),activation='relu',max_iter=5000,random_state=42)
model.fit(X_train,y_train)
prediction=model.predict(X_test)
print(classification_report(y_test,prediction))
