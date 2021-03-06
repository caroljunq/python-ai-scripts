# This algorithm is an implementation of KNN Regression using open modules
#
#
# Supervised Machine Learning
# In Supervised Machine Learning algorithms, you have input variables (x) and output variables (y)
# and you use an algorithm to learn the mapping function from the input to the output Y = f(x).
# This approach aims to approximate the mapping function so well that when you have new input data (x),
# you can predict the output variables (y) for that data.
#
#
# Regression Approach
# Regression is a technique for applying Supervised Machine Learning used when the output is a real or continous value
# (ex: salary, weight). Given a set of inputs and outputs (target), the regression algorithm try to find the best function which
# maps the given inputs to the given outputs. After, for new inputs the algorithm predict the outputs (continous or real values).
# There some ways to implement regression: linear regression, regression trees, support vector regression (SVR)
#
#
# KNN - K Nearest Neighbors - Regression
# KNN is an algorithm for applying regression approach. From one sample and a dataset training,the KNN determines
# the distance (euclidian, manhatan, etc.) from one "no valued sample" to all samples in the training set.
# The second step is to get the K neighbors nearest the "no valued sample" (K smallest distances to the given point).
# With the set of K nearest neighbors, the "no valued sample" is valued with the average of the numerical target (y, outputs or labels)
# of the K neighbors. Therefore, the target value for one sample is determined by the average of the targets of the K neighbors.
#
#
#
# References
# https://medium.com/simple-ai/classification-versus-regression-intro-to-machine-learning-5-5566efd4cb83
# https://www.geeksforgeeks.org/regression-classification-supervised-machine-learning/
# https://medium.com/@adi.bronshtein/a-quick-introduction-to-k-nearest-neighbors-algorithm-62214cea29c7
# http://www.saedsayad.com/k_nearest_neighbors_reg.htm
# https://www.antoniomallia.it/on-implementing-k-nearest-neighbor-for-regression-in-python.html

# datasets example
from sklearn import datasets
# KNN Regression
from sklearn.neighbors import KNeighborsRegressor


# dataset example
# Details - https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html
boston = datasets.load_boston()
x,y = boston.data, boston.target

# knn regression, p =2 --> euclidean distance
knn = KNeighborsRegressor(n_neighbors=11,p=2)

# training
knn.fit(x,y)
# getting output values
outputs = knn.predict(x)
