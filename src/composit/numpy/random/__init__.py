import inspect

import numpy as np

from composit.introspection import get_name_from_args_and_kwargs
from composit.numpy.core import create_input


def random(size):
    name = get_name_from_args_and_kwargs(inspect.currentframe().f_code.co_name, size)

    def initialize_callback():
        return np.random.random(size)

    return create_input(name, initialize_callback, size, np.float32)
