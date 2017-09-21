# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 17:29:55 2017

@author: Satish
"""
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
data = pd.read_csv('dataset_1.csv')
X = data.ix[:, 0:3].values
#y = dataset.ix[:, 13].values

# Standerising the dataset
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_standard = sc.fit_transform(X)

# Calculating the mean and Covariance
mean_value = np.mean(X_standard, axis=0)
cov_matrix = (X_standard - mean_value).T.dot((X_standard - mean_value)) / (X_standard.shape[0]-1)
print('\n Covariance matrix \n%s' %cov_matrix)

##Another way to Calculate Covariance Matrix
cov_matrix = np.cov(X_standard.T)
print('NumPy Covariance matrix: \n%s' %cov_matrix)

eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)

print('\n Eigenvectors \n%s' %eigen_vectors)
print('\n Eigenvalues \n%s' %eigen_values)

## From Singular Vector Decomposition  
#U,S,V = np.linalg.svd(X_standard.T)
#print('\n Vectors U:\n', U)

#for e_vect in eigen_vectors:
#    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(e_vect))
#print('Everything ok!')

# Preparing List of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:,i]) for i in range(len(eigen_values))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda x: x[0], reverse=True)

# Printing the Eigen Values
print('\n Eigenvalues in descending order:')
for i in eigen_pairs:
    print(i[0])

eigen_pairs
print('\n Eigen Pairs \n', eigen_pairs)

# Calculating the Variance Explained    
total = sum(eigen_values)
variance_explained = [(i / total)*100 for i in sorted(eigen_values, reverse=True)]
cum_variance_explained = np.cumsum(variance_explained)

print('\n Variance Explained \n', variance_explained)
print('\n Cumulative Variance Explained \n', cum_variance_explained)

matrix_W = np.hstack((eigen_pairs[0][1].reshape(3,1), 
                      eigen_pairs[1][1].reshape(3,1)))

print('\n Matrix W:\n', matrix_W)

X_Reduced = X_standard.dot(matrix_W)

print('\n X_Reduced:\n', X_Reduced)

pca_1 = X_Reduced[:,0]
pca_2 = X_Reduced[:,1]

print('\n pca_1:\n', pca_1)
print('\n pca_2:\n', pca_2)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(pca_1, pca_2)
fig.show()

