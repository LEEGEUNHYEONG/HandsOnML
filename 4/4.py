# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
'''
    4.1 선형회귀
'''
X = 2 * np.random.rand(100, 1)
Y = 4 + 3 * X + np.random.rand(100, 1)

X_b = np.c_[np.ones((100, 1)), X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(Y)

print(theta_best)

# %%
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]

y_predict = X_new_b.dot(theta_best)
print(y_predict)

plt.plot(X_new, y_predict, 'r-')
plt.plot(X, Y, 'b.')
plt.axis([0, 2, 0, 15])
plt.show()

# %%
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, Y)
print(lin_reg.intercept_, lin_reg.coef_)
print(lin_reg.predict(X_new))

# %%
'''
    4.2 경사 하강법
'''

# %% 4.2.1 배치 경사 하강법
eta = 0.1  # 학습률
n_iterations = 1000
m = 1000
theta = np.random.rand(2, 1)

for iteration in range(n_iterations):
    gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - Y)
    theta = theta - eta * gradients

print(theta)

# %% 4.2.2 확률적 경사 하강법 (SGD)
n_epochs = 50
t0, t1 = 5, 50


def learning_schedule(t):
    return t0 / (t + t1)


theta = np.random.randn(2, 1)

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index: random_index + 1]
        yi = Y[random_index: random_index + 1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
print(theta)

# %% SGDRegressor
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(max_iter=50, penalty=None, eta0=0.1)
sgd_reg.fit(X, Y.ravel())

print(sgd_reg.intercept_, sgd_reg.coef_)

# %% 4.3 다항 회귀
m = 100
x = 6 * np.random.rand(m, 1) - 3
y = 0.5 * x ** 2 + x + 2 + np.random.randn(m, 1)

from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly_features.fit_transform(x)
print(x[0], ":::", x_poly[0])

# %%
lin_reg = LinearRegression()
lin_reg.fit(x_poly, y)
print(lin_reg.intercept_, " , ", lin_reg.coef_)

# %% 4.4 학습 곡선
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def plot_learning_curves(model, x, y):
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
    train_error, val_errors = [], []

    for m in range(1, len(x_train)):
        model.fit(x_train[:m], y_train[:m])
        y_train_predict = model.predict(x_train[:m])
        y_val_predict = model.predict(x_val)
        train_error.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    plt.plot(np.sqrt(train_error), "r-+", linewidth=2, label="test set")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="validation set")
    plt.legend()
    plt.show()


lin_reg = LinearRegression()
plot_learning_curves(lin_reg, x, y)

# %%
from sklearn.pipeline import Pipeline

polynomial_regression = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression()),
])

plot_learning_curves(polynomial_regression, x, y)

# %%
'''
    4.5 규제가 있는 선형 모델
'''

# %% 4.5.1 릿지 회귀
from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(x, y)
print("ridge : ", ridge_reg.predict([[1.5]]))

sgd_reg = SGDRegressor(max_iter=5)
sgd_reg.fit(x, y.ravel())
print("sgd reg : ", sgd_reg.predict([[1.5]]))

# %% 4.5.2 라쏘 회귀
from sklearn.linear_model import Lasso

lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(x, y)
print("lasso reg : ", lasso_reg.predict([[1.5]]))

# %% 4.5.3 엘라스틱넷
from sklearn.linear_model import ElasticNet

elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
elastic_net.fit(x, y)
print("elastic net : ", elastic_net.predict([[1.5]]))

# %% 4.5.4 조기 종료 Early Stopping
'''
    책, github 많이 다름, github이 최선, 그러나 이상함
'''
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
m = 100
x = 6 * np.random.rand(m, 1) - 3
y = 2 + x + 0.5 * x ** 2 + np.random.randn(m, 1)

x_train, x_val, y_train, y_val = train_test_split(x[:50], y[:50].ravel(), test_size=0.5, random_state=10)

x = 9
poly_scalar = Pipeline([
    ("poly_features", PolynomialFeatures(degree=90, include_bias=False)),
    ("std_scaler", StandardScaler())])

x_train_poly_scaled = poly_scalar.fit_transform(x_train)
x_val_poly_scaled = poly_scalar.transform(x_val)

sgd_reg = SGDRegressor(n_iter=1, warm_start=True, penalty=None, learning_rate="constant", eta0=0.0005)

n_epochs = 500
train_errors, val_errors = [], []
for epoch in range(n_epochs):
    sgd_reg.fit(x_train_poly_scaled, y_train)
    y_train_predict = sgd_reg.predict(x_train_poly_scaled)
    y_val_predict = sgd_reg.predict(x_val_poly_scaled)
    train_errors.append(mean_squared_error(y_train, y_train_predict))
    val_errors.append(mean_squared_error(y_val, y_val_predict))

best_epoch = np.argmin(val_errors)
best_val_rmse = np.sqrt(val_errors[best_epoch])

plt.annotate("best model", xy=(best_epoch, best_val_rmse),
             xytext=(best_epoch, best_val_rmse + 1),
             ha="center", arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=16)

best_val_rmse -= 0.03
plt.plot([0, n_epochs], [best_val_rmse, best_val_rmse], linewidth=2)
plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="validation set")
plt.plot(np.sqrt(train_errors), "r--", linewidth=2, label="train set")
plt.legend(loc="upper right", fontsize=14)
plt.xlabel("epoch", fontsize=14)
plt.ylabel("RMSE", fontsize=14)
plt.show()

#%% 4.6.3 결정 경계
from sklearn import datasets
iris = datasets.load_iris()
list(iris.keys())
x = iris["data"][:, 3:]
y = (iris["target"] == 2).astype(np.int)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(solver="liblinear")
log_reg.fit(x, y)

x_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(x_new)
plt.plot(x_new, y_proba[:, 1], "g-", label="iris-virginica")
plt.plot(x_new, y_proba[:, 0], "b--", label="not iris-virginica")
plt.legend()
plt.show()

log_reg.predict([[1.7], [1.5]])

#%% 4.6.4 Softmax Regression
x = iris["data"][:, (2, 3)]
y = iris["target"]

softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C = 10)
softmax_reg.fit(x, y)
softmax_reg.predict([[5, 2]])
softmax_reg.predict_proba([[5, 2]])


