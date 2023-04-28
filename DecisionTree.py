from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# загрузка набора данных
iris = load_iris()
X = iris.data
y = iris.target

# разделение набора данных на обучающие и тестовые данные
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# создание экземпляра классификатора дерева решений
clf = tree.DecisionTreeClassifier(criterion='entropy')

# обучение классификатора на обучающих данных
clf.fit(X_train, y_train)

# оценка качества работы классификатора на тестовых данных
accuracy = clf.score(X_test, y_test)
print("Accuracy:", accuracy)

print(X_test)
y_pred = clf.predict([[6.0, 2.6, 4.5, 1.1], [5.4, 3.6, 1.9, 0.2]])
print("Predicted Class Labels: ", y_pred)
