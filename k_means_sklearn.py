# This algorithm is an implementation of K Means Machine Learning using native modules from python
# (exception sklearn module to get datasets)
#
#
# Unsupervised Machine Learning
# In Unsupervised Machine Learning algorithms, you only have input data (x) and no output.
# The goal of this approach is to model the underlying structure or distribution in the data
# in order to learn more about the data. Therefore, you summarize, group and compressed the dataset.
# Unspervised machine learning algorithms are called unsupervised because you start with "unlabeled data"
# (there's no Y, there's no teacher or correct answers). These algorithms are left to their own devises to discover
# and present the interesting structure in data.
#
# Two type of unsupervised techniques are: clustering and association
#
#
# Clustering
# Clustering technique aims to discover the inherent grouping in the data (group the dataset by similarity).
#  The groups have data points where points in different clusters are dissimilar while points within a cluster are similar
#
#
#
# K-Means - Clustering Algorithm
# K-Means is an algorithm to create k groups (for clustering). A larger k creates smaller groups with more granularity,
# a lower k means larger groups and less granularity. The steps are:
# 1- Choose a number of groups (k);
# 2- Define a centroid (point) for each cluster/group (randomly or using techniques);
# 3- Assign each object to the group that has the closest centroid (ex: euclidean distance can be used);
# 4- When all objects have been assigned, recalculate the positions of the k centroids (calculated as the average position
# of all points in one cluster)
# 5- Repeat steps 3 and 4 until the centroids no longer move.
# This algorithm  is sensiive to the initial randomly selected cluster centres.
#
#
#
# References
# http://stanford.edu/~cpiech/cs221/handouts/kmeans.html
# https://home.deib.polimi.it/matteucc/Clustering/tutorial_html/kmeans.html
# https://machinelearningmastery.com/supervised-and-unsupervised-machine-learning-algorithms/
# https://medium.com/machine-learning-for-humans/unsupervised-learning-f45587588294

# importing modules
from sklearn import datasets
from sklearn.cluster import KMeans

# Loading the Iris dataset
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

x,y = iris_dataset.data, iris_dataset.target

kmeans = KMeans(n_clusters=4)

kmeans.fit(x)

y_means = kmeans.predict(x)

print(y_means)
