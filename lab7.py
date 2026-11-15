import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

df = pd.read_csv("mushrooms.csv")
df.replace('?', np.nan, inplace=True)
mode = df.mode().iloc[0]
df.fillna(mode, inplace=True)
df.drop_duplicates(inplace=True)

y = df['class'].astype('category').cat.codes
X_df = pd.get_dummies(df.drop('class', axis=1), drop_first=True, dtype=int)

X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

X = np.array(X_train).T
Y = np.array(y_train).reshape(1, X.shape[1])
m = X.shape[1]

W1 = np.random.rand(16, X.shape[0]) 
W2 = np.random.rand(8, W1.shape[0]) 
W3 = np.random.rand(1, W2.shape[0]) 

B1 = np.zeros((16, 1))
B2 = np.zeros((8, 1))
B3 = np.zeros((1, 1))

epochs = 5000
alpha = 0.01

xaxis=[]
yaxis=[]

for epoch in range(epochs):
    xaxis.append(epoch)
    Z1 = np.dot(W1, X) + B1
    Y1 = np.maximum(0, Z1)
    
    Z2 = np.dot(W2, Y1) + B2
    Y2 = np.maximum(0, Z2)
    
    Z3 = np.dot(W3, Y2) + B3
    Y3 = 1 / (1 + np.exp(-np.clip(Z3, -500, 500)))
    
    epsilon = 1e-15
    Y3_clipped = np.clip(Y3, epsilon, 1 - epsilon)
    cost = (-1 / m) * np.sum(Y * np.log(Y3_clipped) + (1 - Y) * np.log(1 - Y3_clipped))

    yaxis.append(cost)

    if epoch % 500 == 0:
        print(f"loss={cost}")

    dZ3 = Y3 - Y
    dW3 = (1 / m) * np.dot(dZ3, Y2.T)
    dB3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)

    dZ2 = np.dot(W3.T, dZ3) * (Z2 > 0)
    dW2 = (1 / m) * np.dot(dZ2, Y1.T)
    dB2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.dot(W2.T, dZ2) * (Z1 > 0)
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    dB1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    W1 -= alpha * dW1
    W2 -= alpha * dW2
    W3 -= alpha * dW3
    B1 -= alpha * dB1
    B2 -= alpha * dB2
    B3 -= alpha * dB3

Z1 = np.dot(W1, X_test.T) + B1
Y1 = np.maximum(0, Z1)
    
Z2 = np.dot(W2, Y1) + B2
Y2 = np.maximum(0, Z2)
    
Z3 = np.dot(W3, Y2) + B3
Y3 = 1 / (1 + np.exp(-np.clip(Z3, -500, 500)))

rocx=[]
rocy=[]
y_pred=Y3

ypredt = y_pred.flatten()
ypredt.sort()


for yp in ypredt:
    ytemp=(y_pred>=yp).astype(int).flatten()
    tn, fp, fn, tp = confusion_matrix(y_test, ytemp).ravel()
    recall=tp/(tp+fn)
    specificity = tn / (tn + fp)
    rocy.append(recall)
    rocx.append(1-specificity)


y_pred = (Y3 > 0.5).astype(int).flatten()

print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


print("NEURAL NETWORK (MLPClassifier)")
mlp = MLPClassifier(hidden_layer_sizes=(16, 8), activation='relu', max_iter=500, random_state=42)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_mlp))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_mlp))

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