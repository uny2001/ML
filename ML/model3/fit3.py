import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

iris_df = pd.read_csv("penguins3.csv")
label_encoder = LabelEncoder()
iris_df["island"] = label_encoder.fit_transform(iris_df["island"])
iris_df["species"] = label_encoder.fit_transform(iris_df["species"])
X = iris_df.drop(["sex"], axis=1)
Y = iris_df["sex"]
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X, Y, test_size=0.2, random_state=3)
model = DecisionTreeClassifier(criterion="entropy")
model.fit(X_train1, Y_train1)
tree.plot_tree(model, class_names=True)
class_names = iris_df['sex']
tree.plot_tree(model, class_names=class_names)
with open('Iris_pickle_fileTREE.pkl', 'wb') as pkl:
    pickle.dump(model, pkl)
