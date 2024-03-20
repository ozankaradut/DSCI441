# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 15:35:03 2024

@author: ozan.karadut
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor

path1 = "ANN_Ready_Data_01_16_23.csv"
common_col = ['Al', 'Ca', 'Fe', 'Mg', 'Na', 'Si', 'Ti', 'K', 'C',
              '% Ash DB', 'MAF BTU, DB', '% Sulfur, DB', '% Carbon',
              'SiO2', 'Al2O3', 'Fe2O3', 'CaO', 'MgO', 'K2O', 'Na2O',
              'Initial Temp.', 'Slagging Temp.']

puremsw = pd.read_csv(path1)[common_col].dropna(axis=0, how='any')

# Split the datas for regression
x_train, x_test, y_train, y_test = train_test_split(puremsw.iloc[:, :-13], puremsw.iloc[:, -13:],
                                                                  test_size=0.3, random_state=0)


mlp_reg = MLPRegressor(hidden_layer_sizes=[150, 150, 100, 100, 50], activation = 'relu', random_state = 42)
pipeline = make_pipeline(StandardScaler(), mlp_reg)
pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("RMSE:", rmse)
r2 = r2_score(y_test, y_pred)
print("r2_score:", r2)