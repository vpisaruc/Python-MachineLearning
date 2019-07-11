import numpy as np
import pandas as pd
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split

"""В этом примере мы будем предугадывать злокачественные/доброкачественные опухали.
   Для этого мы используем данные из датасета: breast-cancer-wisconsin.data.
   Данный датасет был взят с сайта: https://archive.ics.uci.edu/ml/datasets.php
   где находятся много датасетов подходящих для тренировки навыков в Машинном Обучении"""


df = pd.read_csv(r'Datasets\breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
# выкидываем id колонку, т.к она не имеет никакого влияния на рак груди и будет портить точность предсказания
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

# данные для предсказания
example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 1, 2, 2, 2, 3, 2, 1]])
example_measures = example_measures.reshape(len(example_measures), -1)

prediction = clf.predict(example_measures)
print(prediction)