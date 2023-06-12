from pyrsistent import immutable
from toolz.functoolz import partial


from composit.numpy.core import create_from_numpy_compute_operation


def wrap_as_operation():
    def outer_wrapper(compute_function):
        def wrapper(*operands, **klass_kwargs):
            klass_attributes = list(klass_kwargs.keys())
            klass = immutable(klass_attributes, name=compute_function.__name__)
            klass.__call__ = partial(compute_function, **klass_kwargs)
            operation = klass(**klass_kwargs)
            return create_from_numpy_compute_operation(*operands, operation=operation)

        return wrapper

    return outer_wrapper


__all__ = [
    "wrap_as_operation",
]
