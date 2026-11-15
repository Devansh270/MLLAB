import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('heart_cleveland_upload.csv')
df.drop(columns=['fbs'], inplace=True)

X = df.drop(columns=['condition']).values
y = df['condition'].values

y = np.where(y <= 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lr = 0.01
C = 1.0
n_iters = 1000

n_samples, n_features = X_train.shape

w = np.zeros(n_features)
b = 0.0

for epoch in range(n_iters):
  for i in range(n_samples):
    score = np.dot(X_train[i], w) + b

    if y_train[i] * score >= 1:
      w = w - lr * w
    else:
      w = w - lr * (w - C * y_train[i] * X_train[i])
      b = b - lr * (-C * y_train[i])

test_scores = np.dot(X_test, w) + b

y_pred = np.where(np.sign(test_scores) <= 0, 0, 1)
y_test_original = np.where(y_test <= 0, 0, 1)

print(f"Accuracy: {accuracy_score(y_test_original, y_pred) * 100:.2f}%")
print(classification_report(y_test_original, y_pred))