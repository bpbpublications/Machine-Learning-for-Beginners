import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_blobs

n_samples = 200
random_state = 10

X, y = make_blobs(n_samples=n_samples, random_state=random_state)
y_predicted = SpectralClustering(n_clusters=3, random_state=random_state).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_predicted)
plt.title("Spectral Clustering")
plt.show()

X_1, y_1 = make_blobs(n_samples=n_samples, cluster_std=[1, 0.5, 3.0], random_state=random_state)
y_predicted = SpectralClustering(n_clusters=3, random_state=random_state).fit_predict(X_1)
plt.scatter(X_1[:, 0], X_1[:, 1], c=y_predicted)
plt.title("Spectral Clustering II")
plt.show()

X_2, y_2 = make_blobs(n_samples=n_samples, cluster_std=[1, 1.5, 2.5], random_state=random_state)
y_pred = SpectralClustering(n_clusters=3, random_state=random_state).fit_predict(X_2)
plt.scatter(X_2[:, 0], X_2[:, 1], c=y_predicted)
plt.title("Spectral Clustring III")
plt.show()

X_not_balanced = np.vstack((X[y == 0][:500], X[y == 1][:200], X[y == 2][:10]))
y_predicted= SpectralClustering(n_clusters=3,random_state=random_state).fit_predict(X_not_balanced)
plt.scatter(X_not_balanced[:, 0], X_not_balanced[:, 1], c=y_predicted)
plt.title("Blobs having unequal number of elements in each cluster")
plt.show()