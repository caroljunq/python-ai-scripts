#
#
# Density-Based Clustering
# Density-Based Clustering is a type of clustering (unsupervised machine learning).
# From a dataset, the algorithm identifies the clusters/groups by detecting areas where
# points are concentrated and where they are separated by areas that area empty or sparse. Points
# that are not part of a cluster are labeled as noise.
# This type of algorithm automatically detects patterns based purely on spatial location and the distance
# to a specified number of neighbors. These algorithms are considered unsupervised because
# they do not require any training on what it means to be a cluster.
#
# Algorithm DBSCAN
# DBSCAN (Density-based spatial clustering of applications with noise) is one algorithm for applying
# the density-based clustering. This method uses a specified distance (search distance)
# to separate dense clusters from sparser noise. Therefore, the basic idea of this algorithm is to group
# together points in high-density.
#
#
#
#
# References
# http://pro.arcgis.com/en/pro-app/tool-reference/spatial-statistics/how-density-based-clustering-works.htm
# https://cse.buffalo.edu/~jing/cse601/fa12/materials/clustering_density.pdf
# http://www.cs.fsu.edu/~ackerman/CIS5930/notes/DBSCAN.pdf
# https://www.dummies.com/programming/big-data/data-science/how-to-create-an-unsupervised-learning-model-with-dbscan/
#
#
#
#
from sklearn.cluster import DBSCAN
# generate samples
from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

dbscan = DBSCAN(eps=0.5, min_samples=5, metric='euclidean',algorithm='auto')

# Testing with Iris dataset
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

# You know there are 3 groups (Setosa, Versicolour, and Virginica), test if dbscan will find 3 clusters
outputs = dbscan.fit_predict(x)

# There are 3 classes 0,1 and -1
print(outputs)

# The PCA (Principal Component Analysis) graph  is a statistical procedure that
# uses an orthogonal transformation to convert a set of observations of possibly
# correlated variables (entities each of which takes on various numerical
# values) into a set of values of linearly uncorrelated variables called
# principal components.  This technique is used to emphasize variation and bring out
# strong patterns in a dataset. It's often used to make data easy to explore and
# visualize. Therefore, PCA reduces the dimensionality of a multivariate data to
# two or three principal components, that can be visualized graphically, with
# minimal loss of information.

# two dimensions
# Reference: https://www.dummies.com/programming/big-data/data-science/how-to-create-an-unsupervised-learning-model-with-dbscan/
pca = PCA(n_components=2).fit(iris_dataset.data)
pca_2d = pca.transform(iris_dataset.data)

# plotting for this dataset example (iris dataset with 3 clusters)
# plotting a scatter graph
for i in range(0, pca_2d.shape[0]):
    if dbscan.labels_[i] == 0:
        c1 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='r',marker='+')
    elif dbscan.labels_[i] == 1:
        c2 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='g',marker='o')
    elif dbscan.labels_[i] == -1:
        c3 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='b',marker='*')

plt.legend([c1, c2, c3], ['Cluster 1', 'Cluster 2', 'Noise'])
plt.title('DBSCAN finds 2 clusters and noise')
plt.show()
