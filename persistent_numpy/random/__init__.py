import inspect

import numpy as np

from persistent_numpy.numpy_functions import _get_name_from_args_and_kwargs, _create_ndarray


def random(*args, **kwargs):
    array = np.random.random(*args, **kwargs)
    name = _get_name_from_args_and_kwargs(inspect.currentframe().f_code.co_name, *args, **kwargs)
    return _create_ndarray(name, array)
