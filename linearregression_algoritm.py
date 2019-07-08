from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

"""Линейная регрессия по сути это прямая проходящая идеально по середине
   всех наших точек(значений), уравнение этой прямой y = mx + b, где
   y - это значения которые мы хотим получить, m - наклон прямой, 
   x - это значения по которым мы получаем y и b - это смещение прямой"""


xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)

# находим наклон и смещение нашей прямой, если что-то не понятно, то это математика 7го класса, гуглите про прямые
def best_fit_slope_and_intercept(xs, ys):
    m = (((mean(xs) * mean(ys)) - mean(xs * ys))/
         (pow(mean(xs), 2) - mean(pow(xs, 2))))
    b = mean(ys) - m * mean(xs)
    return m, b

# найдем квадрат ошибки для нахождения коэффициента детерминации(аккуратности)
# квадрат ошибки - это растояние от нашей линии регрессии до реального значения
def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig)**2)

# вычисляем аккуратность нашей линии или коэффициент детерминации
def coefficient_of_tetermination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr/squared_error_y_mean)


m, b = best_fit_slope_and_intercept(xs, ys)

# находим y-ки нашей линии регрессии
regression_line = [(m * x) + b for x in xs]

# предсказываем y по заданному x
predict_x = 8
predict_y = (m * predict_x) + b

# наш коэффициент детерминации, узнаем насколько наша регрессия хороша на этих данных
determ_coef = coefficient_of_tetermination(ys, regression_line)

# выводим нашу линию регрессии
plt.scatter(predict_x, predict_y, color='g')
plt.scatter(xs, ys)
plt.plot(xs, regression_line)

plt.show()



