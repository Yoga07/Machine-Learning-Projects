import sys
import numpy as np
import matplotlib
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import classification_report, accuracy_score
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt


def run():
    print("Python: {}", format(sys.version))
    print("Numpy: {}", format(np.__version__))
    print("Matplotlib: {}", format(matplotlib.__version__))
    print("Pandas: {}", format(pd.__version__))
    print("Sklearn: {}", format(sklearn.__version__))

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin" \
          ".data "
    name = ["id", "clump_thickness", "uniform_cell_size", "uniform_cell_shape", "marginal_adhesion",
            "single_epithelial_size", "bare_nuclei", "bland_chromatin", "normal_nucleoli", "mitoses", "class"]
    df = pd.read_csv(url, names=name)

    # Pre-process the data
    df.replace('?', -9999, inplace=True)
    print("Columns: ", df.axes[1])
    df.drop(['id'], 1, inplace=True)
    print("SHAPE:", df.shape)
    print(df.loc[698])
    print(df.describe())
    df.hist(figsize=(10, 10))
    scatter_matrix(df, figsize=(10, 10))
    plt.show()
    X = np.array(df.drop(['class'], 1))
    y = np.array(df['class'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    seed = 7
    scoring = 'accuracy'
    models = [('KNN', KNeighborsClassifier(n_neighbors=5)), ('SVM', SVC())]

    results = []
    names = []

    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)


if __name__ == '__main__':
    run()
