# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 17:21:56 2017

@author: Satish
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
data = pd.read_csv('dataset_1.csv')
X = data.ix[:, 0:3].values
#y = dataset.ix[:, 13].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_standard = sc.fit_transform(X)

from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=2)
X_reduced = sklearn_pca.fit_transform(X_standard)

X_reduced