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

# Required modules (python native)
# math functions
import math
# generate random number
import random
# test dataset
from sklearn import datasets
import numpy as np

def euclidean_distance(sample1,sample2):
    # sample1, sample2
    sum = 0
    # get the number of attributes in one sample
    for i in range(len(sample1)):
        sum += math.pow(sample1[i] - sample2[i],2)
    # return the distance between two points
    return math.sqrt(sum)

# K_Means class
class K_Means:
    # constructor
    def __init__(self,samples,K=2,iterations=100):
        self.k = K # number of clusters
        self.samples = samples # samples without output
        self.n_samples = len(samples) # number of samples
        self.n_attributes = len(samples[0]) # number of fields/attributes in each sample
        self.max_iterations = iterations

    # training method
    def train(self):
        # initiating clusters' dict
        # each cluster is a list of index of samples in self.samples
        clusters = {}
        # getting k initial random centroids without repetition
        centroids = random.sample(list(self.samples),self.k)
        prev_centroids = []
        iter = 0
        while iter < self.max_iterations:
            clusters = {x: [] for x in range(self.k)}
            for i in range(self.n_samples):
                distances = []
                # calculating the distances from each sample to each centroid
                for centroid in centroids:
                    # saving distances
                    distances.append(euclidean_distance(self.samples[i],centroid))
                # selecting the closest centroid from one sample
                cluster_index = distances.index(min(distances))
                # add index of a sample in the closest cluster
                clusters[cluster_index].append(i)

            # if the algorithm does not stops, prev_clusters now is equal to the current one
            # centroids[:] is pass by value instead of reference, if you just use prev = centroids
            # you're passing by reference, so prev will always be equal to centroids variable
            prev_centroids = centroids[:]
            # new centroids initiating with 0
            new_centroids = [[0 for _ in range(self.n_attributes)] for _ in range(self.k)]
            # calculate new centroids from the mean of all elements in each cluster
            for i in range(self.k):
                # num of elements in the cluster
                cluster_size = len(clusters[i])
                for sample_index in clusters[i]:
                    for attr_index in range(self.n_attributes):
                        new_centroids[i][attr_index] += self.samples[sample_index][attr_index]
                centroids[i] = [total/cluster_size for total in new_centroids[i]]
            iter += 1
            # if the centroids from the last iteration are equal to the current centroids, the algorithm stops
            # using numpy because one of these list are a numpay array
            if np.array_equal(prev_centroids,centroids):
                break
        print(clusters)
        print("\n\n")
        return clusters


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

x_train, y_train = iris_dataset.data,iris_dataset.target #

# in this case, you know there are 3 groups (Setosa, Versicolour, Virginica)
k_means = K_Means(x_train,3,100)

k_means.train()

# Load Wine Dataset
# Details - https://archive.ics.uci.edu/ml/datasets/wine
# Attributes:
# 1) Alcohol
# 2) Malic acid
# 3) Ash
# 4) Alcalinity of ash
# 5) Magnesium
# 6) Total phenols
# 7) Flavanoids
# 8) Nonflavanoid phenols
# 9) Proanthocyanins
# 10)Color intensity
# 11)Hue
# 12)OD280/OD315 of diluted wines
# 13)Proline
## 3 classes
wine_dataset = datasets.load_wine()

x_wine, y_wine = wine_dataset.data, wine_dataset.target #

# in this case, you know there are 3 class
k_means_wine = K_Means(x_wine,3,250)

k_means_wine.train()
