import sklearn as sk
import quandl
import pandas as pd
import math

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









