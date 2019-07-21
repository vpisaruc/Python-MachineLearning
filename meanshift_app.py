import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import MeanShift
import pandas as pd
from sklearn import preprocessing
style.use('ggplot')

df = pd.read_excel(r'Datasets\titanic.xls')
original_df = pd.DataFrame.copy(df)

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

# массив наших параметров
X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = MeanShift()
clf.fit(X)

# наши лэйблы и центры групп
labels = clf.labels_
cluster_centers = clf.cluster_centers_

# создали колонку в первоначальном DataFrame, заполнили ее nan
original_df['cluster_group'] = np.nan

# заполняем колонку cluster_group
for i in range(len(X)):
    # iloc - используется для построчного индексирования DataFrame
    original_df['cluster_group'].iloc[i] = labels[i]
# кол-во групп
n_clusters_ = len(np.unique(labels))

# создание и заполнение словаря со статистикой выживших людей по группам
survival_rates = {}
for i in range(n_clusters_):
    temp_df = original_df[ (original_df['cluster_group'] == float(i)) ]
    survival_cluster = temp_df[ (temp_df['survived'] == 1) ]
    survival_rate = len(survival_cluster)/len(temp_df)
    survival_rates[i] = survival_rate

print(survival_rates)