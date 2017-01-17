import numpy as np
from scipy.optimize import leastsq


def func(x, p):
    A, k, theta = p
    return A * np.sin(2*np.pi*k*x + theta)

