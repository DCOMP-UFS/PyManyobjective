import numpy as np

def sensitivity_func(y_true, y_pred):
    correct_positive_pred = y_pred[(y_true == 1)].sum()
    total_positive = (y_true == 1).sum()
    return correct_positive_pred / total_positive
