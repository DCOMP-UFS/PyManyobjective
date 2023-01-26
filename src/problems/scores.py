import numpy as np

def sensitivity_func(y_true, y_pred):
    correct_positive_pred = (y_pred[(y_true == 1)] == 1).sum()
    total_positive = (y_true == 1).sum()
    return correct_positive_pred / total_positive

def specificity_func(y_true, y_pred):
    correct_negative_pred = (y_pred[(y_true != 1)] != 1).sum()
    total_negative = (y_true != 1).sum()
    return correct_negative_pred / total_negative
