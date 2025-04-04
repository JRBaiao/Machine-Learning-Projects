# First Machine Learning Method: K-Means Clustering.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Upload a basic dataset for introduction.
# Inserting the dataframe format using pandas.
plt.style.use('bmh')
iris = pd.read_csv('iris.csv')
 
# Testing:
# print(iris.head())

# Data coparision examples:
# sns.boxplot(x='species', y='petal_length', data=iris)
# plt.show()

# Removing the column species for the clustering:
x = iris.iloc[:, [0, 1, 2, 3]].values
# print(x[0:5])

# K-Means Clustering - 3 clusters: 
kmeans = KMeans(n_clusters=3, random_state=16)
y_kmeans = kmeans.fit_predict(x)

# Visualizing the clusters:
plt.scatter(x[y_kmeans==0, 0], x[y_kmeans==0,1], 
            s=100, 
            c='blue',
            label='Iris-setosa')

plt.scatter(x[y_kmeans==1, 0], x[y_kmeans==1,1], 
            s=100, 
            c='red',
            label='Iris-versicolor')

plt.scatter(x[y_kmeans==2, 0], x[y_kmeans==2,1], 
            s=100, 
            c='purple',
            label='Iris-virginica')

plt.scatter(kmeans.cluster_centers_[:,0], 
            kmeans.cluster_centers_[:,1], s=100, 
            c='black', label='Centroids')

plt.legend()
plt.show()
# In the example above, we specified the number of clusters as 3.


""" The Elbow Method """

wcss = [] # Within Clustr Sum of Squares 
for i in range(1,10):
    kmeans= KMeans(n_clusters = i,random_state = 16)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,10), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of cluseters')
plt.ylabel('Within Cluster Sum of Sqaures')
plt.show()