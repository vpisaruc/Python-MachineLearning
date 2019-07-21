import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
style.use('ggplot')

df = pd.read_excel(r'Datasets\titanic.xls')
df.drop(['body', 'name'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)

# конвертируем не числовые данные в числовые
def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        # словарь с уникальными значениями в колонках
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        # проверка данных на их тип(числовой или нет)
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            # данные из колонки
            column_contents = df[column].values.tolist()
            # выбирем уникальные элементы в колонках
            unique_elements = set(column_contents)
            x = 0
            # добавление уникального значения в словарь text_digit_vals
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1

            df[column] = list(map(convert_to_int, df[column]))

    return df

df = handle_non_numerical_data(df)
print(df.head())


