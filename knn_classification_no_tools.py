# This algorithm is an implementation of KNN Classification using native modules from python
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
#
# Classification Approach
# Classification is a technique for applying Supervised Machine Learning used when the output is discrete value or a category.
# Given a set of inputs and outputs (labels), the classification algorithm try to find the best function which maps the given
# inputs to given outputs. After, for new inputs the algorithm predict the outputs.
#
#
#
# KNN - K Nearest Neighbors - Classification
# KNN is an algorithm for applying classification approach. From one sample and a dataset training,the KNN determines
# the distance (euclidian, manhatan, etc.) from "no classified sample" to all samples in the training set. The second step is to get
# the K neighbors nearest the "no classified sample"  (K smallest distances to the given point).
# With the set of K nearest neighbors, the "no classified sample" is classified by a majority vote of its neighbors
# (class most common among its k nearest neighbors).
#
#
# References
# https://medium.com/simple-ai/classification-versus-regression-intro-to-machine-learning-5-5566efd4cb83
# https://www.geeksforgeeks.org/regression-classification-supervised-machine-learning/
# https://medium.com/@adi.bronshtein/a-quick-introduction-to-k-nearest-neighbors-algorithm-62214cea29c7
# http://www.saedsayad.com/k_nearest_neighbors_reg.htm


# Required modules (python native)
# math functions
import math
# test dataset
from sklearn import datasets

# x_train - training dataset
# y_train - labels, outputs
# K = number of neighbors, default value = 3
def knn_classification(x_train,y_train,sample_to_be_classified,K=3):
    # keep the distances from sample to all points in the training set
    distances = {}
    # calculating euclidean distance
    for i in range(len(x_train)):
        soma = 0
        for j in range(len(sample_to_be_classified)): # get the number of attributes in one sample. In the case, target sample
            soma += math.pow(sample_to_be_classified[j] - x_train[i][j],2)
        distances[i] = math.sqrt(soma)

    # getting the nearest K neighbors
    k_neighbors = sorted(distances,key=distances.get)[:K]

    # classifying the sampl_to_be_classified
    # depends on each case, this case has 3 classes (0,1,2)
    counter = {
        0: 0,
        1: 0,
        2: 0
    }

    for index in k_neighbors:
        counter[y_train[index]] += 1

    # majority vote
    classification = sorted(counter,key=counter.get)[-1] # get the last element (bigger)
    return classification

# testing knn classification with Iris dataset
def testing():
    # Load the Iris dataset
    # Details - http://archive.ics.uci.edu/ml/datasets/Iris
    # Attributes:
    # 1. sepal length in cm
    # 2. sepal width in cm
    # 3. petal length in cm
    # 4. petal width in cm
    # 5. class: (output/class/y)
    ####-- Iris Setosa
    ####-- Iris Versicolour
    ####-- Iris Virginica
    iris_dataset = datasets.load_iris()

    x_train, y_train = iris_dataset.data,iris_dataset.target # ".data" gets inputs attributes and ".target" gets the output attributes

    # x_train[0] belongs to class 0, so the algorithm should return 0
    print("First test: ", knn_classification(x_train,y_train,x_train[0],5))

    # testing with a random sample [5.3,2.9,4.7,1.2]
    print("Second test: ", knn_classification(x_train,y_train,[5.3,2.9,4.7,1.2],K=10))


testing()
