import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import OrdinalEncoder, PolynomialFeatures
from sklearn.model_selection import GridSearchCV

# Read in the most recent dataset for Repeatability Analysis and Model Validation
path_svm = "ANN_Ready_Data_01_16_23_Classification.csv"

common_col_svm = ['Al', 'Ca', 'Fe', 'Mg', 'Na', 'Si', 'Ti', 'K', 'C', 'Class']

pureclass = pd.read_csv(path_svm)[common_col_svm].dropna(axis=0, how='any')

x_train_svm, x_test_svm, y_train_svm, y_test_svm = train_test_split(pureclass.iloc[:, :-1],

                                                                    pureclass.iloc[:, -1:], test_size=0.2, random_state=0)
# String data convert to Float Data type

ordinal_encoder = OrdinalEncoder()
y_train_svm = ordinal_encoder.fit_transform(y_train_svm).ravel()
y_test_svm = ordinal_encoder.fit_transform(y_test_svm).ravel()

# Support Vector Machine Classification
# hyperparameter:C.Reducing C, makes the street larger, but it also leads to more margin violations. 

svm_clf = make_pipeline(StandardScaler(), LinearSVC(C=100, dual=True, random_state=42))
svm_clf.fit(x_train_svm, y_train_svm)
#dual=True: Solves the dual optimization problem.
#This is generally more efficient when the number of samples (rows) is less than the number of features (columns).
#dual=False: Solves the primal optimization problem.
#This is generally more efficient when the number of features is less than the number of samples.

y_linear_svm = svm_clf.predict(x_test_svm)
score_svm = svm_clf.decision_function(x_test_svm)
# the scores that the SVM used to make these predictions. 
# These measure the signed between each instance and the decision boundary:

accuracy_linear_svm = accuracy_score(y_test_svm, y_linear_svm)
cm_linear_svm = confusion_matrix(y_test_svm, y_linear_svm)     

print("accuracy_linear_svm: ", accuracy_linear_svm)

# Non-Linear SVM Classification

poly_svm_clf = make_pipeline(PolynomialFeatures(degree=3),
                             StandardScaler(),
                             LinearSVC(C=100, max_iter=10_000, random_state=42))
poly_svm_clf.fit(x_train_svm, y_train_svm)
y_poly_svm = poly_svm_clf.predict(x_test_svm)

accuracy_poly_svm = accuracy_score(y_test_svm, y_poly_svm)
cm_poly_svm = confusion_matrix(y_test_svm, y_poly_svm)     

print("accuracy_poly_svm: ", accuracy_poly_svm)

# Polynomial Kernel

kernel_poly_svm_clf = make_pipeline(StandardScaler(),
                                    SVC(kernel="poly", degree=3, coef0=1, C=100))
# The hyperparameter controls how much the model is influenced coef0 by high-degree terms versus low-degree terms
kernel_poly_svm_clf.fit(x_train_svm, y_train_svm)
y_kernel_poly_svm = kernel_poly_svm_clf.predict(x_test_svm)

accuracy_kernel_poly_svm = accuracy_score(y_test_svm, y_kernel_poly_svm)
cm_kernel_poly_svm = confusion_matrix(y_test_svm, y_kernel_poly_svm)   

print("accuracy_kernel_poly_svm: ", accuracy_kernel_poly_svm)

# Gaussian RBF Kernel

rbf_kernel_svm_clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma=5, C=0.001))

rbf_kernel_svm_clf.fit(x_train_svm, y_train_svm)
y_rbf_kernel_svm_clf = rbf_kernel_svm_clf.predict(x_test_svm)

accuracy_rbf_kernel_svm_clf = accuracy_score(y_test_svm, y_rbf_kernel_svm_clf)
cm_rbf_kernel_svm_clf = confusion_matrix(y_test_svm, y_rbf_kernel_svm_clf) 
print("accuracy_rbf_kernel_svm_clf: ", accuracy_rbf_kernel_svm_clf)

# Define the parameter grid
param_grid = {
    'C': [100, 10, 1, 0.01],
    'gamma': [0.01, 0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly']
}

# Create an SVC model
svm_model = SVC()

# Perform grid search with cross-validation
grid_search = GridSearchCV(svm_model, param_grid, cv=5 )  # 5-fold cross-validation

# Train the model on the training data
grid_search.fit(x_train_svm, y_train_svm)

best_params = grid_search.best_params_

print("best_params:", best_params)














