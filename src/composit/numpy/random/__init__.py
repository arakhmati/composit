import inspect

import numpy as np

from composit.introspection import get_name_from_args_and_kwargs
from composit.numpy.core import create_ndarray


def random(*args, **kwargs):
    array = np.random.random(*args, **kwargs)
    name = get_name_from_args_and_kwargs(inspect.currentframe().f_code.co_name, *args, **kwargs)
    return create_ndarray(name, array)
