# This algorithm is an implementation of KNN Regression using native modules from python
# (exception sklearn module to get datasets)
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
#
#
#
# Required modules (python native)
# math functions
import math
# test dataset
from sklearn import datasets

# Regression Class
class KNNRegression:
    def __init__(self,x,y,K=4):
        self.x = x # input values
        self.y = y # output values
        self.K = K # K neighbors
        self.n_samples = len(x) # number of samples
        self.n_attrs = len(x[0]) # number of attributes in the sample[0] ("how many fields")

    # prediction method
    def predict(self,sample_test):
        distances = {}

        # calculates the distances from sample_test to all samples in the dataset
        for i in range(self.n_samples):
            xi_sum = 0 # sum of the sample
            # calculates the distances based on all atributes in one sample
            # euclidean distance
            for j in range(self.n_attrs):
                xi_sum += math.pow(sample_test[j] - self.x[i][j],2)
            distances[i] = math.sqrt(xi_sum)
        # gets k neighbors keys with smallest distances
        k_neighbors = sorted(distances,key=distances.get)[:self.K]
        # return the average of k_neighbors
        mean = sum([self.y[key] for key in k_neighbors])
        return mean/self.K

# dataset example
# Details - https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html
boston = datasets.load_boston()
x,y = boston.data, boston.target

# testing with test set
knn = KNNRegression(x,y,8)

# expected
print("True value: ", y[28])
# test with random sample
print("Predicted value: ", knn.predict(x[28]))

# expected
print("True value: ", y[5])
# test with random sample
print("Predicted value: ", knn.predict(x[5]))
