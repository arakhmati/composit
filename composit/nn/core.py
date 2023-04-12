from pyrsistent import immutable, PClass
from toolz.functoolz import partial


from composit.multidigraph import MultiDiGraph
from composit.numpy.core import create_from_numpy_compute_instruction
from composit.persistent_array import PersistentArray, Node


class Variable(PClass):
    ...


def wrap_as_instruction():
    def outer_wrapper(compute_function):
        compute_function = staticmethod(compute_function)

        def wrapper(*operands, **klass_kwargs):
            klass_attributes = list(klass_kwargs.keys())
            klass = immutable(klass_attributes, name=compute_function.__name__)
            klass.__call__ = partial(compute_function, **klass_kwargs)
            instruction = klass(**klass_kwargs)
            return create_from_numpy_compute_instruction(*operands, instruction=instruction)

        return wrapper

    return outer_wrapper


__all__ = [
    "Variable",
    "wrap_as_instruction",
]
