import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')


class K_Means:
    # k - колличество групп на которые будут разбиваться данные
    # tol - на сколько может быть смещен центр группы
    # max_iter - максимальное кол-во итераций
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        # словарь центров наших двух групп, т.к k=2
        self.centroids = {}

        # предположим, что центры это первые две точки нашего массива точек
        for i in range(self.k):
            self.centroids[i] = data[i]

        # выполняем цикл, пока не найдены подходящие центры или не достигнуто max_iter
        for i in range(self.max_iter):
            # словарь классификация наших групп
            self.classifications = {}

            # заполняем наш словарь пустыми массивами
            for i in range(self.k):
                self.classifications[i] = []

            # вычисляем дистанцию между центроидами и точкой и смотрим к какому центру ближе точка
            # относим точку к той или инной группе в зависимости от того к какому центру она ближе
            for featureset in data:
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            # значение предыдущих центров
            prev_centroids = dict(self.centroids)

            # перевычисляем центры групп
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)
            optimized = True


            for c in self.centroids:
                original_centroids = prev_centroids[c]
                current_centroid = self.centroids[c]
                # если наш центр превышает допустимое смещение tol, циклимся дальше
                if np.sum((current_centroid - original_centroids)/original_centroids * 100) > self.tol:
                    optimized = False

            # если значение tol не превышенное, выходим из цикла
            if optimized:
                break

    def predict(self, data):
        # вычисляем к какому центру ближе точка и присваиваем этой точке группу
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification



X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11]])

# plt.scatter(X[:, 0], X[:, 1], s=150)
# plt.show()

colors = ['g', 'r', 'c', 'b', 'k']

clf = K_Means()
clf.fit(X)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
                marker='o', color='k', s=150, linewidths=5)

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker='x', color=color, s=150, linewidths=5)

unknowns = np.array([[1, 3],
                    [8, 9],
                    [0, 3],
                    [5, 4],
                    [6, 4]])

for unknown in unknowns:
    classification = clf.predict(unknown)
    plt.scatter(unknown[0], unknown[1], color=colors[classification], marker='*', s=150, linewidths=5)

plt.show()
