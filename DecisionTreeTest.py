import pandas as pd
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# загрузка набора данных
dataset = pd.read_csv('data/Cancer_Data.csv')

print(dataset.shape)
print(dataset.columns)

dataset1 = dataset.drop(["id","Unnamed: 32"],axis = 'columns')

X = dataset1.drop('diagnosis',axis='columns')
y = dataset1['diagnosis']

print(dataset1.shape)
print(dataset1.columns)

# разделение набора данных на обучающие и тестовые данные
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# создание экземпляра классификатора дерева решений
clf = tree.DecisionTreeClassifier(criterion='entropy')

# обучение классификатора на обучающих данных
clf.fit(X_train, y_train)

# оценка качества работы классификатора на тестовых данных
accuracy = clf.score(X_test, y_test)
print("Accuracy:", accuracy)

# print(X_test)
y_pred = clf.predict([[17.86, 10.55, 118.8, 1005, 0.1166, 0.2779, 0.3051, 0.1491, 0.2499, 0.07811, 1.035, 0.9058, 8.586, 151.6, 0.006319, 0.04931, 0.05359, 0.01566, 0.03016, 0.006109, 24.68, 17.49, 183.8, 2019, 0.1661, 0.6682, 0.7107, 0.2668, 0.4611, 0.1168],
                      [17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189],
                      [12.54, 13.36, 82.46, 526.3, 0.09979, 0.08229, 0.06864, 0.04581, 0.1985, 0.05966, 0.2688, 0.7986, 3.058, 29.56, 0.007462, 0.0186, 0.02587, 0.01715, 0.0215, 0.0029, 15.51, 19.56, 106.7, 721.6, 0.168, 0.1738, 0.242, 0.1298, 0.2959, 0.0756]])
print("Predicted Class Labels: ", y_pred)
