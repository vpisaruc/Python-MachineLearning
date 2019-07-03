import sklearn as sk
import quandl, math
import pandas as pd
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

"""Linear regression"""
df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

# значения по которым мы пытаемся предугадать значения
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

# столбец который мы пытаемся предугадать
forecast_col = 'Adj. Close'

# Машинное обучение может работать с NaN значениями, если им присвоенно какоето значение
# в нашем случаем - 99999 или нужно избавиться от NaN(не советую, т.к можете потярть много данных)
df.fillna(-99999, inplace=True)

# ceil - округление в большую сторону
forecast_out = int(math.ceil(0.01 * len(df)))

# shift - смещение значений колонки
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

# features или значения по которым мы пытаемся предугадать значения
X = np.array(df.drop(['label'], 1))
# our labels или значения которыу мы пытаемся предугадать
y = np.array(df['label'])

# маштабируем наши данные
X = preprocessing.scale(X)
df.dropna(inplace=True)
y = np.array(df['label'])

# функция train_test_split разбивает наш масивв данных на 4 подмассива,
# 2 из них для обучения нашей систему, 2 для тестов
# test_size = 0.2 - 20% от наших данных
X_train,  X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# определяем объект линейной регрессии
clf = LinearRegression()
# тренируем нашу систему
clf.fit(X_train, y_train)
# считаем насколько наша система хорошо предсказываает значения
accuracy = clf.score(X_test, y_test)

print(accuracy)





