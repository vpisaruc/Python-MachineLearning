import numpy as np
import warnings
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter

style.use('fivethirtyeight')


# считаем евклидово расстояние, не знаете, что это, гуглите
dataset = {'k' : [[1, 2], [2, 3], [3, 1]], 'r' : [[6, 5], [7, 7], [8, 6]]}
new_features = [5, 7]



def k_nearest_neighbours(data, predict, k=3):
    if len(data) > k:
        warnings.warn('K is set to a value less thab total voting groups!')
    distances = []
    for group in data:
        for features in data[group]:
            # один из способов написания евклидовго расстояния
            #euclidean_distance = np.sqrt(np.sum((np.array(features) - np.array(predict))**2))
            # однако этот быстрее
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])

    # записываем 3 ближайшие точки
    votes = [i[1] for i in sorted(distances)[:3]]
    # смотрим к какой группе принадлежит большинство, к этой же группе и будет принадлежать наша точка
    vote_result = Counter(votes).most_common(1)[0][0]

    return vote_result

result = k_nearest_neighbours(dataset, new_features, k=3)

[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0], new_features[1], s=100, color=result)
plt.show()