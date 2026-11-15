import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv('electricity_cost_dataset.csv')

df_final = df[['site area', 'structure type', 'water consumption',
               'utilisation rate','resident count', 'electricity cost']].copy()

featuress = df_final.columns.drop(['structure type', 'electricity cost'])

scaler = StandardScaler()
df_final[featuress] = scaler.fit_transform(df_final[featuress])

df_final = pd.get_dummies(df_final, columns=['structure type'], dtype=int)

X = df_final.drop('electricity cost', axis=1)
y = df_final['electricity cost']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X1 = np.array(X_train).T
Y1 = np.array(y_train)
m = X1.shape[1]

W = np.random.rand(1, X1.shape[0])
B = 0
alpha = 0.01
cost=0
epochs = 2000
yaxis=[]
xaxis=[]
for epoch in range(epochs):
    y_hat = np.dot(W, X1) + B
    
    error = y_hat - Y1
    cost = (1 / m) * np.sum(error ** 2)
    yaxis.append(cost)
    xaxis.append(epoch)
    
    dB = (2 / m) * np.sum(error)
    dW = (2 / m) * np.dot(error, X1.T) 
    
    W -= alpha * dW
    B -= alpha * dB
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Cost: {cost:.4f}")
test=np.array(X_test).T
y_hat=np.dot(W,test)+B
y_hat=y_hat.flatten()
r2_score1=r2_score(y_test,y_hat)
print(f"r2 score(manual):{r2_score1}")
print(f"MSE:{cost}")
print(f"RMSE(Manual): {np.power(cost,0.5)}")

plt.figure(figsize=(8,6))
sns.lineplot(x=xaxis,y=yaxis,color='red',markers='o',linewidth=2)
plt.ylabel("Error")
plt.xlabel("Epochs")
plt.show()


model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("--- Model Evaluation ---")
print(f"R-squared (R²): {r2:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"RMSE:{np.power(mse,0.5)}")