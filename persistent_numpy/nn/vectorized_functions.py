import math

import numpy as np


def erf(input_tensor):
    return np.vectorize(math.erf)(input_tensor)


def cdf(input_tensor):
    return 0.5 * (1 + erf(input_tensor / np.sqrt(2)))


def pdf(input_tensor):
    return 0.3989422804014327 * np.exp(input_tensor * input_tensor * -0.5)
