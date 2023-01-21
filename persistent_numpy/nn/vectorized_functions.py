import math

import numba
import numpy as np


@numba.vectorize(["float64(float64)", "float32(float32)"])
def erf(input_tensor):
    return math.erf(input_tensor)


@numba.vectorize(["float64(float64)", "float32(float32)"])
def cdf(input_tensor):
    return 0.5 * (1 + erf(input_tensor / np.sqrt(2)))


@numba.vectorize(["float64(float64)", "float32(float32)"])
def pdf(input_tensor):
    return 0.3989422804014327 * np.exp(input_tensor * input_tensor * -0.5)
