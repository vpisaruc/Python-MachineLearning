from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11]])

# X[:, 0] такие конструкции выводят n-мерную матрицу по столбцам
# plt.scatter(X[:, 0], X[:, 1], s=150, linewidths=5)
# plt.show()

# n_clusters - число групп на которое мы разбиваем наши данные
clf = KMeans(n_clusters=2)
clf.fit(X)

# координаты центров каждой группы
centroids = clf.cluster_centers_
# опознаватель группы, целове число
# например у нас 2 группы тогда label будет равен 0 или 1
labels = clf.labels_

colors = ['g.', 'r.', 'c.', 'b.', 'k.']

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=25)

plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=150, linewidths=10)
plt.show()
