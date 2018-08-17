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
#
#
#
# Required modules (python native)
# math functions
import math
import random
# test dataset
from sklearn import datasets

class KNNRegression:
    # constructor
    def __init__(self,inputs,outputs,learning_rate=0.1,epochs=1000,bias=-1):
        self.data = inputs # samples for training
        self.target = outputs # outputs for training
        self.learning_rate = learning_rate
        self.epochs = epochs # max number of interactions
        self.bias = bias
        self.total_samples = len(inputs) # the size of samples
        self.n_attrs = len(inputs[0]) # number of attributes in a sample ("how many fields")
        self.weights = [] # weights to adjustments

    def train(self):
        # começa aqui

    def operating(self):
        #começa aqui
