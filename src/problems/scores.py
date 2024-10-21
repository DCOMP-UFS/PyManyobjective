import numpy as np

def sensitivity_func(y_true, y_pred):
    correct_positive_pred = (y_pred[(y_true == 1)] == 1).sum()
    total_positive = (y_true == 1).sum()
    if total_positive == 0:
        print("TOTAL POSITIVE 0")
        return 0
    return correct_positive_pred / total_positive

def specificity_func(y_true, y_pred):
    correct_negative_pred = (y_pred[(y_true != 1)] != 1).sum()
    total_negative = (y_true != 1).sum()
    if total_negative == 0:
        print("TOTAL NEGATIVE 0")
        return 0
    return correct_negative_pred / total_negative 

def precision_func(y_true, y_pred):
    correct_positive_pred = (y_pred[(y_true == 1)] == 1).sum()
    false_positive_pred = (y_pred[(y_true != 1)] == 1).sum()
    if correct_positive_pred + false_positive_pred == 0:
        print("SOMA DEU 0")
        return 0
    return max(0, correct_positive_pred / (correct_positive_pred + false_positive_pred))
