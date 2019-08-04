# %%
import numpy as np

np.random.seed(42)

# %% 6.1 결정 트리 학습과 시각화
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
x = iris.data[:, 2:]
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(x, y)

# %%
from sklearn.tree import export_graphviz

export_graphviz(tree_clf, out_file="6/iris_tree.dot",
                feature_names=iris.feature_names[2:],
                class_names=iris.target_names,
                rounded=True, filled=True)

# %% path 에러 발생 함 !!!
import graphviz

with open("6/iris_tree.dot") as f:
    dot_graph = f.read()
dot = graphviz.Source(dot_graph)
dot.format = 'png'
dot.render(filename="iris_tree.png", directory='6/', cleanup=True)
dot

# %% 6.3 클래스 확률 추정
print(tree_clf.predict_proba([[5, 1.5]]))
print(tree_clf.predict([[5, 1.5]]))

# %% 6.8 회귀
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(x, y)

