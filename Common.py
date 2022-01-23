import numpy as np
import pandas as pd
import scipy.stats as stats


# Возвращает сумму квадратов регрессии
# y_pred - оценка, полученная на основе регрессионной модели
# y_mean - среднее по всем наблюдениям y
def SSR(y_pred: np.array, y_mean: float):
    return np.sum(np.square(y_pred - y_mean))


# Возвращает сумму квадратов ошибок, приходящаяся на одну степень свободы
# y - истинное значение результирующей переменной
# y_pred - оценка, полученная на основе регрессионной модели
# n - изначальный объем выборки
# k - количество уже выбранных переменных
def MSE(y: np.array, y_pred: np.array, n: int, k: int):
    SSE = np.sum(np.square(y - y_pred))
    return SSE / (n - k - 2)


# Возвращает значение частного F-критерия
# y - истинное значение результирующей переменной
# y_pred_initial - оценка, полученная на основе регрессионной модели
#                  без новой переменной
# y_pred_full - оценка, полученная на основе регрессионной модели
#               с добавлением новой переменной
# n - изначальный объем выборки
# k - количество уже выбранных переменных
def calculate_f_real(y: np.array,
                     y_pred_initial: np.array,
                     y_pred_full: np.array,
                     n: int,
                     k: int):
    y_mean = np.mean(y)
    SSR_initial = SSR(y_pred_initial, y_mean) if y_pred_initial is not None else 0
    SSR_full = SSR(y_pred_full, y_mean)
    SSR_extra = SSR_full - SSR_initial
    MSE_full = MSE(y, y_pred_full, n, k)
    return SSR_extra / MSE_full


# расчет граничной величины критерия f_table
# alpha - риск принятия неправильного решения
# n - изначальный объем выборки
# k - количество уже выбранных переменных
def calculate_f_table(alpha: float, n: int, k: int):
    return stats.f.ppf(1 - alpha, 1, n - k - 2)


# Возвращает имя переменной, которая имеет наибольшую корреляцию с y
# X - выборка
# y - истинное значение результирующей переменной
def max_corr_feature(X: pd.DataFrame, y: pd.Series):
    corrs = X.apply(lambda feature: feature.corr(y))
    max_corr_feature_name = corrs.nlargest(1).index[0]
    print(max_corr_feature_name)
    return max_corr_feature_name


# примеры
#
# print(f_table(0.05, 11, 1))
#
# print(f_real(
#     np.array([0, 0, 0, 1, 1, 0, 3, 2, 2, 4]),
#     None,
#     np.array([0.436, 0.436, 0.436, 0.436, 1.396, 0.436, 1.396, 1.396, 2.356, 4.275]),
#     10,
#     0))