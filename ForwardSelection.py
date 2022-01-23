import math
import pandas as pd
from Common import calculate_f_table, calculate_f_real, find_max_corr_feature


class ForwardSelection:
    def __init__(self, regression):
        self.regression = regression

    # X - выборка
    # y - истинное значение результирующей переменной
    def _predict(self, X: pd.DataFrame, y: pd.Series):
        # Для корректной работы с выборкой из одной переменной
        if X.shape[1] == 1:
            X = X.values.reshape(-1, 1)
        self.regression.fit(X, y)
        y_pred = self.regression.predict(X)
        return y_pred

    # Выбор переменной, имеющей максимальное значение f_real
    # X_remains - выборка, среди которой выбираем переменную
    # X_current - уже выбранные переменные
    # y - истинное значение результирующей переменной
    # y_pred_initial - оценка, полученная на основе регрессионной модели
    #                  без новой переменной
    # n - изначальный объем выборки
    # k - количество уже выбранных переменных
    def find_feature_with_max_f_real(self,
                                     X_remains: pd.DataFrame,
                                     X_current: pd.DataFrame,
                                     y: pd.Series,
                                     y_pred_initial: pd.Series,
                                     n: int,
                                     k: int):
        max_f_real = -math.inf
        current_feature_name = None
        for feature_name in X_remains:
            feature = X_remains[feature_name]
            # Расчет значения f_real на выборке X_new, включающей переменную feature
            X_new = X_current.copy()
            X_new[feature_name] = feature
            y_pred_full = self._predict(X_new, y)
            f_real = calculate_f_real(y, y_pred_initial, y_pred_full, n, k)
            if f_real > max_f_real:
                max_f_real = f_real
                current_feature_name = feature_name
        return current_feature_name

    # Отбор значимых переменных из выборки Х
    # X - выборка
    # y - истинное значение результирующей переменной
    # alpha - риск принятия неправильного решения
    def select(self, X: pd.DataFrame, y: pd.Series, alpha: float = 0.1):
        # результат - отобранные переменные
        X_result = pd.DataFrame()

        # выборка, из которой будем проводить отбор переменных
        X_remains = X.copy()

        # объем выборки
        n = X.shape[1]

        # количество выбранных переменных
        k = 0

        y_pred_initial = None

        # выбор переменной, которая имеет наибольшую корреляцию с y
        current_feature_name = find_max_corr_feature(X, y)
        X_current = pd.DataFrame()

        for _ in range(n):
            if current_feature_name is None:
                break
            # добавляем новую переменную в текущую выборку
            X_current[current_feature_name] = X_remains[current_feature_name]

            # расчет значения критерия f_real
            y_pred_full = self._predict(X_current, y)
            f_real = calculate_f_real(y, y_pred_initial, y_pred_full, n, k)
            # расчет граничной величины критерия f_table
            f_table = calculate_f_table(alpha, n, k)

            if f_real > f_table:
                # переменная current_feature - значимая
                # включение переменной current_feature в результирующую выборку
                X_result = X_current.copy()
                y_pred_initial = y_pred_full
                k += 1
                # удаляем текущую переменную из дальнейшего рассмотрения
                X_remains.pop(current_feature_name)
            else:
                # больше значимых переменных нет, конец отбора
                break

            # выбор новой переменной
            current_feature_name = self.find_feature_with_max_f_real(
                X_remains, X_current, y, y_pred_initial, n, k)

        return X_result
