import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

df = pd.read_csv('heart_cleveland_upload.csv')
df.drop_duplicates(inplace=True)
df.drop(columns=['fbs'], inplace=True)

X = df.drop(columns=['condition'])
y = df['condition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = SVC(kernel='linear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(' Scikit-Learn Linear SVM Report ')
print(classification_report(y_test, y_pred))

X1 = X_train
Y1 = np.where(y_train <= 0, -1, 1)

n_samples, n_features = X1.shape
W = np.zeros(n_features)
B = 0.0

alpha = 0.01
c = 5
epochs = 1000

for epoch in range(epochs):
  for i in range(n_samples):
    score = np.dot(X1[i], W) + B

    if Y1[i] * score >= 1:
      W = W - alpha * W
    else:
      W = W - alpha * (-W - c * Y1[i] * X1[i])
      B = B - alpha * (-c * Y1[i])

y_predd = np.dot(X_test, W) + B
y_pred_custom = np.where(y_predd <= 0, 0, 1)

print('Custom Linear SVM Report')
print(classification_report(y_test, y_pred_custom))

final_scores = np.dot(X1, W) + B
support_vector_mask = Y1 * final_scores <= 1
custom_support_vectors = X1[support_vector_mask]
print(f'Number of custom support vectors: {len(custom_support_vectors)}')

gamma = 0.1
C_svm = 5.0
learning_rate = 0.001
max_iter = 10000
tol = 1e-5

y_train_svm = np.where(Y1 == 0, -1.0, 1.0)

sq_dists_train = (
    np.sum(X1**2, axis=1, keepdims=True)
    + np.sum(X1**2, axis=1)[None, :]
    - 2 * X1 @ X1.T
)

sq_dists_train = np.maximum(sq_dists_train, 0)

K_train = np.exp(-gamma * sq_dists_train)

Q = np.outer(y_train_svm, y_train_svm) * K_train

n_samples = len(y_train_svm)
alphas = np.zeros(n_samples)

def project_alpha(v, y, C):
    low, high = -np.max(np.abs(v)) - C - 1, np.max(np.abs(v)) + C + 1

    for _ in range(100):
        mid = (low + high) / 2
        projected = np.clip(v - mid * y, 0, C)

        if np.dot(y, projected) > 0:
            low = mid
        else:
            high = mid

    return np.clip(v - ((low + high) / 2) * y, 0, C)


for iteration in range(max_iter):
    gradient = np.ones(n_samples) - Q @ alphas

    new_alphas = project_alpha(
        alphas + learning_rate * gradient,
        y_train_svm,
        C_svm
    )

    change = np.max(np.abs(new_alphas - alphas))
    alphas = new_alphas

    if change < tol:
        break

support_mask = (alphas > 1e-5) & (alphas < C_svm - 1e-5)

if np.any(support_mask):
    indices = np.where(support_mask)[0]

    b_values = (
        y_train_svm[indices]
        - K_train[np.ix_(indices, np.arange(n_samples))]
        @ (alphas * y_train_svm)
    )

    b_svm = np.mean(b_values)
else:
    b_svm = 0.0

sq_dists_test = (
    np.sum(X_test**2, axis=1, keepdims=True)
    + np.sum(X1**2, axis=1)[None, :]
    - 2 * X_test @ X1.T
)

sq_dists_test = np.maximum(sq_dists_test, 0)

K_test = np.exp(-gamma * sq_dists_test)

decision_values = K_test @ (alphas * y_train_svm) + b_svm
y_rbf = np.where(decision_values > 0, 1, 0)

print("Custom Gradient-Descent RBF-SVM Report")
print(classification_report(y_test, y_rbf, zero_division=0))
print(f"Iterations: {iteration + 1}")
print(f"Number of support vectors: {np.sum(alphas > 1e-5)}")

rbf_model = SVC(kernel='rbf', C=5, gamma='scale')
rbf_model.fit(X_train, y_train)
y_pred_rbf = rbf_model.predict(X_test)

print('Scikit-Learn RBF SVM Report')
print(classification_report(y_test, y_pred_rbf))


mod = MLPClassifier(
    hidden_layer_sizes=(16, 8), activation='relu', max_iter=500, random_state=42
)
mod.fit(X_train, y_train)
ypredict = mod.predict(X_test).flatten()
print('MLP Report')
print(classification_report(ypredict, y_test))