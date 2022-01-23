import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from ForwardSelection import ForwardSelection

# Загружаем экземпляр набора данных, вызвав функцию load_boston()

boston = load_boston()

df = pd.DataFrame(boston.data, columns=boston.feature_names)

df['PRICE'] = pd.Series(boston.target)

# Выбираем первые 13 столбцов в качестве переменных
X = df.iloc[:, :13]

# Выбираем колонку Price как целевое значение
y = df['PRICE']

# Выбор значимых переменных, alpha - риск принятия неправильного решения
selection = ForwardSelection(LinearRegression())
X = selection.select(X, y, alpha=0.1)
print(X.columns.values)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

lin_reg_mod = LinearRegression()
lin_reg_mod.fit(X_train, y_train)
y_pred = lin_reg_mod.predict(X_test)

test_set_rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))
print(test_set_rmse)

test_set_r2 = r2_score(y_test, y_pred)
print(test_set_r2)
