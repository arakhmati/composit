import math

import numpy as np


def embedding(input_tensor: np.array, weights: np.array) -> np.array:
    batch_size, sequence_size = input_tensor.shape
    _, hidden_size = weights.shape
    result = np.zeros((batch_size, sequence_size, hidden_size), dtype=weights.dtype)
    for batch_index in range(batch_size):
        for sequence_index in range(sequence_size):
            result[batch_index, sequence_index] = weights[input_tensor[batch_index, sequence_index]]
    return result


def erf(input_tensor: np.array) -> np.array:
    return np.vectorize(math.erf)(input_tensor)


def cdf(input_tensor: np.array) -> np.array:
    return (0.5 * (1 + erf(input_tensor / np.sqrt(2)))).astype(input_tensor.dtype)


def pdf(input_tensor: np.array) -> np.array:
    return (0.3989422804014327 * np.exp(input_tensor * input_tensor * -0.5)).astype(input_tensor.dtype)


def gelu(input_tensor: np.array) -> np.array:
    return input_tensor * cdf(input_tensor)
