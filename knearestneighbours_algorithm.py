import numpy as np
import warnings
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import pandas as pd
import random

style.use('fivethirtyeight')


# считаем евклидово расстояние, не знаете, что это, гуглите
# dataset = {'k' : [[1, 2], [2, 3], [3, 1]], 'r' : [[6, 5], [7, 7], [8, 6]]}
#new_features = [5, 7]

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
    confidence = Counter(votes).most_common(1)[0][1] / k

    return vote_result, confidence

accurasies = []

for i in range(5):
    df = pd.read_csv(r'Datasets\breast-cancer-wisconsin.data')
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], 1, inplace=True)
    full_data = df.astype(float).values.tolist()
    # смешаем данные
    random.shuffle(full_data)

    test_size = 0.4
    train_set = {2:[], 4:[]}
    test_set = {2:[], 4:[]}
    train_data = full_data[:-int(test_size*len(full_data))]
    test_data = full_data[-int(test_size*len(full_data)):]

    for i in train_data:
        train_set[i[-1]].append(i[:-1])

    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    correct = 0
    total = 0

    for group in test_set:
        for data in test_set[group]:
            vote, confidence = k_nearest_neighbours(train_set, data, k=5)
            if group == vote:
                correct += 1

            total += 1

    accurasies.append(correct/total)

print(sum(accurasies)/len(accurasies))

# result = k_nearest_neighbours(dataset, new_features, k=3)

# [[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
# plt.scatter(new_features[0], new_features[1], s=100, color=result)
# plt.show()