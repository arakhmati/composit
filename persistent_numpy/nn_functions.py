import inspect
import math

import numpy as np
from pyrsistent import immutable

from persistent_numpy.numpy_functions import _create_from_numpy_compute_instruction


def embedding(*operands):
    def compute(self, input_tensor, weights):
        batch_size, sequence_size = input_tensor.shape
        result = np.zeros((batch_size, sequence_size, weights.shape[1]))
        for batch_index in range(batch_size):
            for sequence_index in range(sequence_size):
                result[batch_index, sequence_index] = weights[input_tensor[batch_index, sequence_index]]
        return result

    function_name = inspect.currentframe().f_code.co_name
    klass = immutable(name=function_name)
    klass.__call__ = compute
    instruction = klass()
    return _create_from_numpy_compute_instruction(*operands, instruction=instruction)


def gelu(operand):
    def compute(self, input_tensor):
        return 0.5 * input_tensor * (1 + np.vectorize(math.erf)(input_tensor / np.sqrt(2)))

    function_name = inspect.currentframe().f_code.co_name
    klass = immutable(name=function_name)
    klass.__call__ = compute
    instruction = klass()
    return _create_from_numpy_compute_instruction(operand, instruction=instruction)
