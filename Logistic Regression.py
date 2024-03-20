import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import confusion_matrix

# Read in the most recent daataset for Repeatability Analysis and Model Validation
path_cs = "ANN_Ready_Data_01_16_23_Classification.csv"
common_col_cs = ['Al', 'Ca', 'Fe', 'Mg', 'Na', 'Si', 'Ti', 'K', 'C', 'Class']
pureclass = pd.read_csv(path_cs)[common_col_cs].dropna(axis=0, how='any')

# Split the datas for regression
pure_train_cs, pure_test_cs, pure_ytrain_cs, pure_ytest_cs = train_test_split(pureclass.iloc[:, :-1],
                                                                        pureclass.iloc[:, -1:], test_size=0.3,
                                                                              random_state=42)
ordinal_encoder = OrdinalEncoder()
pure_ytrain_cs = ordinal_encoder.fit_transform(pure_ytrain_cs).ravel()
pure_ytest_cs = ordinal_encoder.fit_transform(pure_ytest_cs).ravel()

X_train_logr = pure_train_cs
X_test_logr = pure_test_cs
y_train_logr = pure_ytrain_cs
y_test_logr = pure_ytest_cs

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_logr, y_train_logr)
y_pred_logr = log_reg.predict(X_test_logr)

cm_1 = confusion_matrix(y_test_logr, y_pred_logr)
accuracy = np.trace(cm_1) / np.sum(cm_1)
print("Logistic Regression Accuracy:", accuracy)

softmax_reg = LogisticRegression(C=1,random_state=42)
softmax_reg.fit(X_train_logr, y_train_logr)
y_pred_softmax = softmax_reg.predict(X_test_logr)

cm_2 = confusion_matrix(y_test_logr, y_pred_softmax)
accuracy = np.trace(cm_2) / np.sum(cm_2)
print("Softmax Regression Accuracy:", accuracy)


