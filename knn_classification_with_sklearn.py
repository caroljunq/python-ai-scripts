# This algorithm is an implementation of KNN Classification using sklearn and numpy modules


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
# KNN - K Nearest Neighbors
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

# sklearn it is a package dedicated to machine learning
# datasets = include some datasets example for testing
from sklearn import datasets
# KNeighborsClassifier = knn algorithm/classifier in sklearn package
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

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
iris = datasets.load_iris()

# x = inputs values
# y = output values/labels/classes
x,y = iris.data, iris.target

# When you have a dataset, you can use part of the data for training and the other part for testing
# x_train = input values for training
# y_train = output values for training
# x_test = input values for testing
# y_test = output values for testing
# The test dataset will have 30% (0.3) of the data - In the case of Iris dataset 150 registers, 30% = 45 samples
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

# KNeighborsClassifier is the class of the knn algorithm
# n_neighbors = k neighbors
# p = type of distance to be used, p = 2 is the euclidian distance
knn = KNeighborsClassifier(n_neighbors=7,p=2)
# training with the training dataset
knn.fit(x_train,y_train)
# testing the classifier with the testing dataset
outputs = knn.predict(x_test)

# If you want to check how well the classifier got success classifying the testing dataset
# you can use score function
#in this case the score was 1.0 (all - 100%)
print("Test 1: ",knn.score(x_test,y_test))


# testing with breast cancer dataset
# Details: http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29
cancer = datasets.load_breast_cancer()

# x_cancer = input values
# y_cancer = output values
x_cancer, y_cancer = cancer.data, cancer.target
x_train2,x_test2,y_train2,y_test2 = train_test_split(x_cancer,y_cancer,test_size=0.2,random_state=42)
knn2 = KNeighborsClassifier(n_neighbors=8,p=2)
knn2.fit(x_train2,y_train2)
# classification for cancer testing dataset
labels = knn2.predict(x_test2)

# in this case 0.956140350877193 - 95,6% of success on classifying this testing dataset
print("Test 2: ",knn2.score(x_test2,y_test2))
