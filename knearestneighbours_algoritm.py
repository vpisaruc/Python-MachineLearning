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

# [[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
# plt.scatter(new_features[0], new_features[1], s=100, color='g')
# plt.show()

def k_nearest_neighbours(data, predict, k=3):
    if len(data) > k:
        warnings.warn('K is set to a value less thab total voting groups!')
    knnalgos
    return vote_result