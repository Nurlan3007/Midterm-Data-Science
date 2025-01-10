import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv('datasets/abalone.data', header=None)
df.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight',
              'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']

df = pd.get_dummies(df, drop_first=True)

X = df.drop('Rings', axis=1)
y = df['Rings']

X = sm.add_constant(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Предсказанные значения:")
print(y_pred[:10])

def bias_variance_curve(model, X_train, X_test, y_train, y_test, n_iter=100):
    train_errors = []
    test_errors = []

    for _ in range(n_iter):
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        train_errors.append(mean_squared_error(y_train, y_train_pred))

        y_test_pred = model.predict(X_test)
        test_errors.append(mean_squared_error(y_test, y_test_pred))

    mean_train_error = np.mean(train_errors)
    mean_test_error = np.mean(test_errors)

    return mean_train_error, mean_test_error


train_error, test_error = bias_variance_curve(model, X_train, X_test, y_train, y_test)

print(f"Ошибка на тренировочных данных: {train_error}")
print(f"Ошибка на тестовых данных: {test_error}")

errors = [train_error, test_error]
labels = ['Train', 'Test']

plt.plot(labels, errors, marker='o')
plt.title("Bias-Variance Curve")
plt.xlabel("Model Complexity")
plt.ylabel("Error")
plt.grid(True)
plt.show()

plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)  # Линия y = x
plt.xlabel("Реальные значения (True Values)")
plt.ylabel("Предсказанные значения (Predicted Values)")
plt.title("Реальные vs Предсказанные значения")
plt.show()
