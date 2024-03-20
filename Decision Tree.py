# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:24:39 2024

@author: ozan.karadut
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OrdinalEncoder


# Read in the most recent dataset for Repeatability Analysis and Model Validation
path = "ANN_Ready_Data_01_16_23_Classification.csv"

common_col = ['Al', 'Ca', 'Fe', 'Mg', 'Na', 'Si', 'Ti', 'K', 'C', 'Class']

pureclass = pd.read_csv(path)[common_col].dropna(axis=0, how='any')

x_train, y_train = pureclass.iloc[:, :-1],pureclass.iloc[:, -1:]

tree_clf = DecisionTreeClassifier(max_depth=None, 
                                  min_samples_split=8,#Min # of samples a node must have before it can be split
                                  min_samples_leaf=5, #Min # of samples a leaf node must have to be created
                                  max_features=1, #Max # of features that are evaluated for splitting at each node
                                  max_leaf_nodes=None, #max # of leaf nodes
                                  criterion='gini', #entropy
                                  random_state=42)

tree_clf.fit(x_train, y_train)

# Plot the decision tree
plt.figure(figsize=(15, 10))
plot_tree(
    tree_clf,
    feature_names= x_train.columns.tolist() ,
    class_names=y_train.iloc[:, -1].unique().tolist(),  # Convert array to list
    filled=True,
    rounded=True
)
plt.show()


# Regression Tree

ordinal_encoder = OrdinalEncoder()
y_train_reg = ordinal_encoder.fit_transform(y_train).ravel()


tree_reg = DecisionTreeRegressor(max_depth = None,
                                 min_samples_split = 2,
                                 min_samples_leaf = 5, 
                                 max_features = 10, 
                                 max_leaf_nodes = None, 
                                 criterion='squared_error', 
                                 random_state=42
                                )

tree_reg.fit(x_train, y_train_reg)

# Plot the regression-decision tree
plt.figure(figsize=(15, 10))
plot_tree(
    tree_reg,
    feature_names= x_train.columns.tolist() ,
    class_names=y_train.iloc[:, -1].unique().tolist(),  # Convert array to list
    filled=True,
    rounded=True
)
plt.show()
