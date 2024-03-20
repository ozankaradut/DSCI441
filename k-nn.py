
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Read in the most recent daataset for Repeatability Analysis and Model Validation
path_cs = "ANN_Ready_Data_01_16_23_Classification.csv"
common_col_cs = ['Al', 'Ca', 'Fe', 'Mg', 'Na', 'Si', 'Ti', 'K', 'C', 'Class']
pureclass = pd.read_csv(path_cs)[common_col_cs].dropna(axis=0, how='any')

x_train, x_test, y_train, y_test = train_test_split(pureclass.iloc[:, :-1], pureclass.iloc[:, -1:],
                                                                  test_size=0.3 , random_state=42)


k_values = [1,2,3,4,5]
accuracy_values = []

for k in k_values:
                knn_clf = KNeighborsClassifier(n_neighbors=k)
                knn_clf.fit(x_train, y_train.values.ravel())
                y_pred = knn_clf.predict(x_test).ravel()
                #Measure Accuracy
                cm = confusion_matrix(y_test, y_pred)
                accuracy = np.trace(cm) / np.sum(cm)
                print("Accuracy for k =", k, ":", accuracy)
                accuracy_values.append(accuracy)
