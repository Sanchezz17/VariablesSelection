import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from ForwardSelection import ForwardSelection
from BackwardElimination import BackwardElimination

# Загружаем экземпляр набора данных, вызвав функцию load_boston()

boston = load_boston()

print(boston.keys())

df = pd.DataFrame(boston.data, columns=boston.feature_names)

print(boston.DESCR)

df['PRICE'] = boston.target

# Выбираем первые 13 столбцов в качестве переменных
X = df.iloc[:, :13]

# Выбираем колонку Price как целевое значение
y = df['PRICE']


def regression_test(X: pd.DataFrame, y: pd.Series):
    # Выбор значимых переменных, alpha - риск принятия неправильного решения
    print(X.columns.values)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred = lin_reg.predict(X_test)

    rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))
    print(rmse)

    r2 = r2_score(y_test, y_pred)
    print(r2)

print("\nForward Selection test\n")
selection = ForwardSelection(LinearRegression())
X_selected = selection.select(X, y, alpha=0.1)
regression_test(X_selected, y)

print("\nBackward Elimination test\n")
selection = BackwardElimination(LinearRegression())
X_selected = selection.select(X, y, alpha=0.01)
regression_test(X_selected, y)
