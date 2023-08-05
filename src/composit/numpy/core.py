from __future__ import annotations

import inspect

import numpy as np
from pyrsistent import immutable, PClass, field
from toolz import functoolz, memoize, partial

from composit.introspection import class_name
from composit.multidigraph import MultiDiGraph, merge_graphs
from composit.types import LazyTensor, Node


def get_operands(graph, node):
    def sort_key(in_edge):
        _, _, data = in_edge
        return data["sink_input_index"]

    def node_operand(in_edge):
        predecessor, _, data = in_edge
        return predecessor, data["source_output_index"]

    return functoolz.pipe(
        graph.in_edges(node, data=True),
        list,
        functoolz.partial(sorted, key=sort_key),
        functoolz.partial(map, node_operand),
    )


@functoolz.memoize
def operation_shape_and_dtype(operation, input_shapes_and_dtypes):
    dummy_input_arrays = [np.zeros(input_shape, dtype=dtype) for input_shape, dtype in input_shapes_and_dtypes]
    result = operation(*dummy_input_arrays)

    if np.isscalar(result):
        result = np.asarray(result)

    if isinstance(result, np.ndarray):
        return (result.shape,), (result.dtype,)
    elif isinstance(result, (list, tuple)):
        return tuple(array.shape for array in result), tuple(array.dtype for array in result)
    else:
        raise RuntimeError(f"Unsupported type: {type(result)}")


class Input(PClass):
    initializer_callback = field()

    def __call__(self, *input_arrays: list[np.ndarray]):
        return self.initializer_callback()

    def __hash__(self):
        # only node name matters
        return 0


def create_input(name: str, initializer_callback, shape, dtype):
    node = Node(name=name)
    graph = MultiDiGraph().add_node(
        node, operation=Input(initializer_callback=initializer_callback), shapes=(shape,), dtypes=(np.dtype(dtype),)
    )
    return LazyTensor(graph=graph, node=node)


def preprocess_operands(*operands):
    operands = list(operands)
    for index, operand in enumerate(operands):
        if not isinstance(operand, (int, float)):
            continue

        dtype = operands[0].dtype
        array = np.asarray(operand, dtype)

        def initializer_callback():
            return array

        operands[index] = create_input(
            f"scalar_{operand}",
            initializer_callback,
            (),
            dtype,
        )
    return operands


def compute_node_hash(operation, *operands):
    if isinstance(operation, PClass):
        operation_hash = tuple(operation._to_dict())
    else:
        operation_hash = operation._fields

    operand_hashes = [(operand.name, operand.output_index) for operand in operands]
    operand_hashes = tuple(operand_hashes)

    return hash((operation_hash, operand_hashes))


def create_from_numpy_compute_operation(
    *operands,
    operation,
    dtype_to_override=None,
) -> LazyTensor | tuple[LazyTensor]:
    operands = preprocess_operands(*operands)

    graph = merge_graphs(*tuple((operand.graph, operand.node) for operand in operands))

    shapes, inferred_dtypes = operation_shape_and_dtype(
        operation,
        tuple((operand.shape, operand.dtype) for operand in operands),
    )
    dtypes = tuple(dtype_to_override or inferred_dtype for inferred_dtype in inferred_dtypes)

    node_hash = compute_node_hash(operation, *operands)
    name = f"{class_name(operation)}_{node_hash}"
    new_node = Node(name=name)
    graph = graph.add_node(new_node, operation=operation, shapes=shapes, dtypes=dtypes)
    for index, operand in enumerate(operands):
        graph = graph.add_edge(
            operand.node,
            new_node,
            source_output_index=operand.output_index,
            sink_input_index=index,
        )

    result = tuple(
        LazyTensor(graph=graph, node=new_node, output_index=output_index) for output_index, dtype in enumerate(dtypes)
    )
    if len(result) == 1:
        return result[0]
    return result


@memoize
def create_numpy_compute_operation(function_name, *args, **kwargs):
    numpy_function = getattr(np, function_name)
    if isinstance(numpy_function, np.ufunc):
        klass_kwargs = {}
    else:
        numpy_function_parameters = inspect.signature(numpy_function).parameters

        klass_args = [key for key, value in numpy_function_parameters.items() if value.default == inspect._empty]
        klass_kwargs = {
            key: value.default for key, value in numpy_function_parameters.items() if value.default != inspect._empty
        }

        # Use passed in kwargs
        for key, value in kwargs.items():
            klass_kwargs[key] = value

        # Distribute args across klass_args and klass_kwargs
        for index, arg in enumerate(args):
            if isinstance(arg, LazyTensor):
                continue
            if index < len(klass_args):
                key = klass_args[index]
                klass_kwargs[key] = arg
            else:
                kwargs_index = len(klass_args) - index
                key = list(klass_kwargs.keys())[kwargs_index]
                klass_kwargs[key] = arg

    def compute(self, *args, **kwargs):
        return numpy_function(*args, **kwargs, **self._asdict())

    klass_attributes = list(klass_kwargs.keys())
    klass = immutable(klass_attributes, name=function_name)
    klass.__call__ = compute

    return klass(**klass_kwargs)


def create_numpy_compute_function(function_name):
    def function(*args, **kwargs):
        operands = [arg for arg in args if isinstance(arg, LazyTensor)]
        return create_from_numpy_compute_operation(
            *operands,
            dtype_to_override=kwargs.get("dtype", None),
            operation=create_numpy_compute_operation(function_name, *args, **kwargs),
        )

    return function


def create_numpy_binary_compute_function(function_name):
    def function(*args, **kwargs):
        operand_a, operand_b, *args = args
        return create_from_numpy_compute_operation(
            operand_a,
            operand_b,
            dtype_to_override=kwargs.get("dtype", None),
            operation=create_numpy_compute_operation(function_name, *args, **kwargs),
        )

    return function


def create_numpy_concatenate_function():
    @memoize
    def create_concatenate_operation(axis=0, out=None, dtype=None, casting="same_kind"):
        klass_kwargs = {
            "axis": axis,
            "out": out,
            "dtype": dtype,
            "casting": casting,
        }

        def compute(self, *inputs):
            return np.concatenate(inputs, **self._asdict())

        klass_attributes = list(klass_kwargs.keys())
        klass = immutable(klass_attributes, name="concatenate")
        klass.__call__ = compute

        return klass(**klass_kwargs)

    def function(inputs, *args, **kwargs):
        return create_from_numpy_compute_operation(
            *inputs,
            dtype_to_override=kwargs.get("dtype", None),
            operation=create_concatenate_operation(*args, **kwargs),
        )

    return function


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
    "create_input",
    "create_from_numpy_compute_operation",
    "create_numpy_compute_function",
    "create_numpy_binary_compute_function",
    "create_numpy_concatenate_function",
    "get_operands",
    "wrap_as_operation",
]
