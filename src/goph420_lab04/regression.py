import numpy as np

def multi_regress(y, Z):
    """ Perform multiple linear regression.

    Parameters
    ----------
    y : array_like, shape = (n,) or (n, 1)
        The vector of dependent variable data
    Z : array_like, shape = (n,m)
        The matrix of independent variable data

    Returns
    -------
    numpy.ndarray, shape = (m,) or (m,1)
        The vector of model coefficients
    numpy.ndarray, shape = (n,) or (n,1)
        The vector of residuals
    float
        The coefficient of determination, r^2
    """
    #converting y to a column vector.
    y = np.asarray(y).reshape(-1, 1)
    Z = np.asarray(Z)
    #similar to the example in class, creating a column of 1s and horizontally stacking it to Z to compute the coefficients.
    Z = np.hstack([np.ones((Z.shape[0], 1)), Z.reshape(-1, 1)])
    #computing the coefficients according to the equation in lecture notes and lab manual.
    coefficients = np.linalg.inv(Z.T @ Z) @ Z.T @ y
    #computing the residuals.
    e = y - (Z @ a)
    #computing the coefficient of determination.
    r_squared = 1 - (np.sum(e**2)/np.sum((y - np.mean(y))**2))

    #returning the arrays and flattening them to ensure the correct shape is output - a one dimensional array.
    return coefficients.flatten(), e.flatten(), r_squared