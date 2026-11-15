import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("mushrooms.csv")
df.replace('?', np.nan, inplace=True)
mode = df.mode().iloc[0]
df.fillna(mode, inplace=True)

df.drop_duplicates(inplace=True)

y = df['class'].astype('category').cat.codes
X = pd.get_dummies(df.drop('class', axis=1), drop_first=True, dtype=int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X1 = np.array(X_train).T
Y1 = np.array(y_train).reshape(1, X1.shape[1])

W = np.random.rand(1, X1.shape[0])
B = 0
alpha = 0.1
epochs = 5000
xaxis=[]
yaxis=[]
for epoch in range(epochs):
    xaxis.append(epoch)
    temp = np.dot(W, X1) + B
    yhat = 1 / (1 + np.exp(-temp))
    
    epsilon = 1e-15
    yhat = np.clip(yhat, epsilon, 1 - epsilon)
    cost = -np.mean(Y1 * np.log(yhat) + (1 - Y1) * np.log(1 - yhat))
    yaxis.append(cost)
    
    dW = (1 / X1.shape[1]) * np.dot((yhat - Y1), X1.T)
    dB = (1 / X1.shape[1]) * np.sum(yhat - Y1)
    
    W -= alpha * dW
    B -= alpha * dB
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss {cost:.4f}")

temp = np.dot(W, np.array(X_test).T) + B
ypred = 1 / (1 + np.exp(-temp))

rocx=[]
rocy=[]

ypredt = ypred.flatten()
ypredt.sort()

for yp in ypredt:
    ytemp=(ypred>=yp).astype(int).flatten()
    tn, fp, fn, tp = confusion_matrix(y_test, ytemp).ravel()
    recall=tp/(tp+fn)
    specificity = tn / (tn + fp)
    rocy.append(recall)
    rocx.append(1-specificity)


ypred=(ypred>=0.5).astype(int).flatten()

print(confusion_matrix(y_test,ypred))
print(classification_report(y_test,ypred))



model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

print("Library")
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

plt.figure(figsize=(8,6))
sns.lineplot(x=xaxis,y=yaxis,color='red',markers='o',linewidth=2)
plt.ylabel("Error")
plt.xlabel("Epochs")
plt.show()

plt.figure(figsize=(8,6))
sns.lineplot(x=rocx,y=rocy,color='red',markers='o',linewidth=2)
plt.ylabel("Recall")
plt.xlabel("1-Specificity")
plt.show()