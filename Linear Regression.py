import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve
from sklearn.linear_model import SGDRegressor


# Read in the most recent daataset for Repeatability Analysis and Model Validation
path1 = "ANN_Ready_Data_01_16_23.csv"
common_col = ['Al', 'Ca', 'Fe', 'Mg', 'Na', 'Si', 'Ti', 'K', 'C',
              '% Ash DB', 'MAF BTU, DB', '% Sulfur, DB', '% Carbon',
              'SiO2', 'Al2O3', 'Fe2O3', 'CaO', 'MgO', 'K2O', 'Na2O',
              'Initial Temp.', 'Slagging Temp.']
puremsw = pd.read_csv(path1)[common_col].dropna(axis=0, how='any')

# Split the datas for regression
pure_train, pure_test, pure_ytrain, pure_ytest = train_test_split(puremsw.iloc[:, :-13], puremsw.iloc[:, -13:],
                                                                  test_size=0.2, random_state=0)
X_train = pure_train
X_test = pure_test
y_train = pure_ytrain
y_test = pure_ytest


# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
#print("Bias term: ", "\n", lin_reg.intercept_, "\n", "Weight: ", "\n", lin_reg.coef_, "\n")

RMSE = mean_squared_error(y_test, lin_reg.predict(X_test))
rmse = np.sqrt(RMSE)
print("RMSE: ", rmse)
score = lin_reg.score(X_test, y_test)
print("\nModel Score (R^2):", score, "\n")

# Learning Curve use cross validation technique
Xlc_train, ylc_train, = (puremsw.iloc[:, :-13], puremsw.iloc[:, -13:])

train_sizes, train_scores, valid_scores = learning_curve(
    LinearRegression(), Xlc_train, ylc_train, train_sizes=np.linspace(0.01, 1, 100), cv=5,
    scoring="neg_root_mean_squared_error")
train_errors = -train_scores.mean(axis=1)
valid_errors = -valid_scores.mean(axis=1)

plt.figure(figsize=(5, 3))  
plt.plot(train_sizes, train_errors, "r-+", linewidth=2, label="train")
plt.plot(train_sizes, valid_errors, "b-", linewidth=3, label="valid")

plt.xlabel("Training set size")
plt.ylabel("RMSE")
plt.grid()
plt.legend(loc="upper right")
plt.show()

# Normalization
std_scalar = StandardScaler()
X_train_std_scaled = std_scalar.fit_transform(X_train)
X_test_std_scaled = std_scalar.fit_transform(X_test)
y_train_std_scaled = std_scalar.fit_transform(y_train)
y_test_std_scaled = std_scalar.fit_transform(y_test)

lin_reg.fit(X_train_std_scaled, y_train_std_scaled)
RMSE = mean_squared_error(y_test_std_scaled, lin_reg.predict(X_test_std_scaled))
rmse = np.sqrt(RMSE)
print("Normalized RMSE: ", rmse)
score = lin_reg.score(X_test_std_scaled, y_test_std_scaled)
print("\nModel Score with Normalized DataS (R^2):", score, "\n")

# Logarithmic Input
x_train_log = np.log(X_train)
x_test_log = np.log(X_test)

lin_reg = LinearRegression()
lin_reg.fit(x_train_log, y_train)
RMSE = mean_squared_error(y_test, lin_reg.predict(x_test_log))
rmse = np.sqrt(RMSE)
print("RMSE with logarithmic value of inputs: ", rmse)
score = lin_reg.score(x_test_log, y_test)
print("\nModel Score with Logarithmic Datas (R^2):", score, "\n")

# Square Root of Input
x_train_sqrt = np.sqrt(X_train)
x_test_sqrt = np.sqrt(X_test)

lin_reg = LinearRegression()
lin_reg.fit(x_train_sqrt, y_train)
RMSE = mean_squared_error(y_test, lin_reg.predict(x_test_sqrt))
rmse = np.sqrt(RMSE)
print("RMSE with square root of inputs: ", rmse)
score = lin_reg.score(x_test_sqrt, y_test)
print("\nModel Score (R^2) with sqrt datas:", score, "\n")

# Create an SGDRegressor model
# sgd_reg = SGDRegressor(max_iter=1000, tol=1e-5, penalty=None, eta0=0.01,n_iter_no_change=100, random_state=42)
# sgd_reg.fit(X_train, y_train[: 1])
# sgd_reg.intercept_, sgd_reg.coef_
