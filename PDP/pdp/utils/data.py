import numpy as np


def dict_apply(x, func):
    result = dict()
    for key, value in x.items():
       
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        elif (isinstance(value, str) or isinstance(value, list)):
            result[key] = value
        elif (isinstance(value, np.int64)):
            result[key] = value 
        else:
            result[key] = func(value)
    return result