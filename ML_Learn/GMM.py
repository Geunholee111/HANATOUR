from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

iris_data = load_iris()
irisDF = pd.DataFrame(data=iris_data.data, columns=['sepal_length','sepal_width','petal_length','petal_width'])
print(irisDF)

gmm = GaussianMixture(n_components=3, random_state=0).fit(iris_data.data)
gmm_cluster_labels = gmm.predict(iris_data.data)
print(gmm_cluster_labels)
irisDF['gmm_cluster'] = gmm_cluster_labels
irisDF['target'] = iris_data.target
print(irisDF)

iris_result = irisDF.groupby(['target'])['gmm_cluster'].value_counts()
print("iris_result : \n", iris_result)



pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(iris_data.data)

irisDF['pca_x'] = pca_transformed[:,0]
irisDF['pca_y'] = pca_transformed[:,1]
print(irisDF)

maker0_ind = irisDF[irisDF['gmm_cluster']==0].index
maker1_ind = irisDF[irisDF['gmm_cluster']==1].index
maker2_ind = irisDF[irisDF['gmm_cluster']==2].index

plt.scatter(x=irisDF.loc[maker0_ind,'pca_x'], y=irisDF.loc[maker0_ind, 'pca_y'], marker='o')
plt.scatter(x=irisDF.loc[maker1_ind,'pca_x'], y=irisDF.loc[maker1_ind, 'pca_y'], marker='s')
plt.scatter(x=irisDF.loc[maker2_ind,'pca_x'], y=irisDF.loc[maker2_ind, 'pca_y'], marker='^')

plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('3 Cluster Visualization by 2 PCA Components')
plt.show()
123123123124