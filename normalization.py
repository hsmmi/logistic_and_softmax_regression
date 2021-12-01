import numpy as np

def zero_mean_unit_variance(mat):
    """
    Z-score normalization:
    Get a matrix(or vector) and return zero-mean unit-variance matrix(or vector)
    If you decrease scaler from all elements, the variance doesn't change so first
    decrease all elements by their column mean and then divide it by their column
    variance
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        return (mat - mat.mean(axis=0)) / mat.std(axis=0)

def range_min_to_max(mat, newMin, newMax):
    """
    Scaling to a range:
    Get a matrix(or vector) and return matrix(or vector) that scale all value
    to the range of newMin to newMax
    It's good when:
    .   You know the approximate upper and lower bounds on your data with 
        few or no outliers.
    .   Your data is approximately uniformly distributed across that range.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        return (mat - mat.min(axis=0)) / (mat.max(axis=0) - mat.min(axis=0)) * (newMax - newMin) + newMin

def clipping(mat, newMin, newMax):
    """
    Clipping to a range:
    Get a matrix(or vector) and return matrix(or vector) that set values more than 
    newMax to newMax and values less than newMin to newMin
    It's good when:
    .   When the feature contains some extreme outliers.
    """
    return np.clip(mat,newMin,newMax)

def log_scaling(mat):
    """
    Log Scaling:
    Get a matrix(or vector) and return ) that each element is
    the log of that element in input matrix(or vector
    It's good when:
    .   When the feature contains some extreme outliers.
    """
    return np.log(mat)