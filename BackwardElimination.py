import math
import pandas as pd
from Common import calculate_f_table, calculate_f_real


class BackwardElimination:
    def __init__(self, regression):
        self.regression = regression

    # X - независимые переменные
    # y - истинное значение результирующей переменной
    def _predict(self, X: pd.DataFrame, y: pd.Series):
        # Для корректной работы с Х, состоящим из одной переменной
        if X.shape[1] == 1:
            X = X.values.reshape(-1, 1)
        self.regression.fit(X, y)
        y_pred = self.regression.predict(X)
        return y_pred

    # Возвращает имя переменной, имеющей минимальное значение f_real
    #   и само минимальное значение f_real
    # X - переменные, среди которых выбираем
    # y - истинное значение результирующей переменной
    # y_pred_full - оценка, полученная на основе регрессионной модели с X
    # n - объем выборки
    # k - количество уже выбранных переменных
    def find_feature_with_min_f_real(self,
                                     X: pd.DataFrame,
                                     y: pd.Series,
                                     y_pred_full: pd.Series,
                                     n: int,
                                     k: int) -> (str, float):
        min_f_real = math.inf
        min_f_real_feature_name = None
        for feature_name in X:
            # Расчет значения f_real на переменных X_new, не включающих
            # переменную feature
            X_new = X.copy()
            X_new.pop(feature_name)
            y_pred_initial = self._predict(X_new, y)
            f_real = calculate_f_real(y, y_pred_initial, y_pred_full, n, k)
            if f_real < min_f_real:
                min_f_real = f_real
                min_f_real_feature_name = feature_name
        return min_f_real_feature_name, min_f_real

    # Отбор значимых переменных из Х
    # X - независимые переменные
    # y - истинное значение результирующей переменной
    # alpha - риск принятия неправильного решения
    def select(self, X: pd.DataFrame, y: pd.Series, alpha: float = 0.1):
        # переменные, из которой будем проводить исключение
        X_result = X.copy()

        # объем выборки
        n = X.shape[0]

        # количество выбранных переменных
        k = X.shape[1]

        y_pred_full = self._predict(X, y)

        for _ in range(n):
            # выбор наименее полезной для модели переменной
            min_f_real_feature_name, min_f_real = self.find_feature_with_min_f_real(
                X_result, y, y_pred_full, n, k)

            # расчет граничной величины критерия f_table
            f_table = calculate_f_table(alpha, n, k)

            if min_f_real <= f_table:
                # переменная min_f_real_feature_name - не значимая
                # удаление переменной min_f_real_feature_name из модели
                X_result.pop(min_f_real_feature_name)
                y_pred_full = self._predict(X_result, y)
                k -= 1
            else:
                # больше не значимых переменных нет, конец исключения
                break

        return X_result
