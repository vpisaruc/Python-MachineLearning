from statistics import mean
import numpy as np
import matplotlib.pyplot as plt

"""Линейная регрессия по сути это прямая проходящая идеально по середине
   всех наших точек(значений), уравнение этой прямой y = mx + b, где
   y - это значения которые мы хотим получить, m - наклон прямой, 
   x - это значения по которым мы получаем y и b - это смещение прямой"""


xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)

# находим наклон нашей прямой, если что-то не понятно, то это математика 7го класса, гуглите про прямые
def best_fit_slope(xs, ys):
    m = (((mean(xs) * mean(ys)) - mean(xs * ys))/
         (pow(mean(xs), 2) - mean(pow(xs, 2))))
    return m

m = best_fit_slope(xs, ys)

print(m)

