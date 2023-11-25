from contextlib import contextmanager
import math

from loguru import logger
import numpy as np
import torch
import torch.nn.functional

import composit as cnp
from flashlight.introspection import convert_torch_tensors_to_lazy_tensors, reset_graph_input_index
from flashlight import Tensor

TORCH = torch.__dict__.copy()
TORCH_TENSOR = {attr: getattr(torch.Tensor, attr) for attr in dir(torch.Tensor)}
TORCH_NN_FUNCTIONAL = torch.nn.functional.__dict__.copy()

TORCH_ATTRIBUTES_TO_LEAVE_AS_IS = [
    "arange",
    "argmax",  # TODO: override?
    "autograd",
    "backends",
    "batch_norm",  # TODO: override?
    "bfloat16",
    "bool",
    "BoolTensor",
    "_C",
    "cdouble",
    "cfloat",
    "complex32",
    "device",
    "distributed",
    "double",
    "dtype",
    "embedding",  # TODO: override?
    "embedding_renorm_",  # TODO: override?
    "empty",
    "finfo",
    "float",
    "float16",
    "float32",
    "FloatTensor",
    "_from_functional_tensor",
    "from_numpy",
    "full",  # TODO: override?
    "full_like",  # TODO: override?
    "fx",
    "Generator",
    "get_default_dtype",
    "group_norm",  # TODO: override?
    "int",
    "int8",
    "int16",
    "int32",
    "int64",
    "isfinite",
    "_is_functional_tensor",
    "is_grad_enabled",
    "is_tensor",
    "jit",
    "_jit_internal",
    "layer_norm",  # TODO: override?
    "long",
    "LongTensor",
    "masked_select",
    "max_pool2d",  # TODO: override?
    "nn",
    "no_grad",
    "ones",
    "optim",
    "overrides",
    "rand",  # TODO: override?
    "randint",  # TODO: override?
    "randn",  # TODO: override?
    "relu",  # TODO: override?
    "relu_",  # TODO: override?
    "save",
    "set_grad_enabled",
    "Size",
    "sparse_csr",
    "sparse_csc",
    "sparse_bsr",
    "sparse_bsc",
    "strided",
    "tensor",
    "Tensor",
    "Size",
    "SymFloat",
    "SymInt",
    "_tensor_str",
    "torch",
    "uint8",
    "utils",
    "__version__",
    "where",
    "zeros",
    "zeros_like",
]


TORCH_TENSOR_ATTRIBUTES_TO_LEAVE_AS_IS = {
    "__bool__",  # TODO: override?
    "__class__",
    "__contains__",  # TODO: override?
    "__dict__",
    "__dir__",
    "__eq__",  # TODO: override?
    "__getattribute__",
    "__gt__",  # TODO: override?
    "__hash__",
    "__init__",
    "__index__",  # TODO: override?
    "__le__",  # TODO: override?
    "__lt__",  # TODO: override?
    "__new__",
    "__repr__",
    "__setattr__",
    "__torch_function__",
    "__torch_dispatch__",
    "any",  # TODO: override?
    "add",  # TODO: override?
    "argmax",  # TODO: override?
    "as_subclass",
    "data",
    "detach",  # TODO: potentially override?
    "device",
    "dim",
    "div",  # TODO: override?
    "dtype",
    "eq",  # TODO: override?
    "fill_",  # TODO: override?
    "gt",  # TODO: override?
    "grad_fn",  # TODO: potentially override?
    "is_floating_point",
    "item",  # TODO: override?
    "le",  # TODO: override?
    "lt",  # TODO: override?
    "_make_subclass",
    "masked_fill_",  # TODO: override?
    "normal_",  # TODO: override?
    "mul",  # TODO: override?
    "ne",  # TODO: override?
    "normal_",  # TODO: override?
    "numel",
    "numpy",
    "new",
    "prod",  # TODO: override?
    "repeat",  # TODO: override?
    "size",
    "shape",
    "softmax",  # TODO: override?
    "sub",  # TODO: override?
    "tile",  # TODO: override?
    "tolist",
    "truediv",  # TODO: override?
    "type_as",  # TODO: override?
    "uniform_",  # TODO: override?
    "zero_",  # TODO: override?
}

TORCH_NN_FUNCTIONAL_ATTRIBUTES_TO_LEAVE_AS_IS = [
    "handle_torch_function",
    "has_torch_function_unary",
    "has_torch_function_variadic",
    "_list_with_default",
    "_no_grad_embedding_renorm_",
    "torch",
    "_verify_batch_size",
]


def identity(run_torch):
    def implementation(*args, **kwargs):
        (lazy_input,) = convert_torch_tensors_to_lazy_tensors(*args)
        if run_torch:
            output, *_ = args
            return Tensor(output, lazy_input)
        else:
            return lazy_input

    return implementation


def add(run_torch):
    def implementation(*args):
        lazy_input_a, lazy_input_b = convert_torch_tensors_to_lazy_tensors(*args, allow_scalars=True)
        lazy_output = lazy_input_a + lazy_input_b
        if run_torch:
            output = TORCH_TENSOR["__add__"](*args)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def radd(run_torch):
    def implementation(*args):
        lazy_input_b, lazy_input_a = convert_torch_tensors_to_lazy_tensors(*args, allow_scalars=True)
        lazy_output = lazy_input_a + lazy_input_b
        if run_torch:
            output = TORCH_TENSOR["__radd__"](*args)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def sub(run_torch):
    def implementation(*args):
        lazy_input_a, lazy_input_b = convert_torch_tensors_to_lazy_tensors(*args, allow_scalars=True)
        lazy_output = lazy_input_a - lazy_input_b
        if run_torch:
            output = TORCH_TENSOR["__sub__"](*args)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def rsub(run_torch):
    def implementation(*args):
        lazy_input_a, lazy_input_b = convert_torch_tensors_to_lazy_tensors(*args, allow_scalars=True)
        lazy_output = lazy_input_b - lazy_input_a
        if run_torch:
            output = TORCH_TENSOR["__sub__"](*args)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def mul(run_torch):
    def implementation(*args):
        lazy_input_a, lazy_input_b = convert_torch_tensors_to_lazy_tensors(*args, allow_scalars=True)
        lazy_output = lazy_input_a * lazy_input_b
        if run_torch:
            output = TORCH_TENSOR["__mul__"](*args)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def rmul(run_torch):
    def implementation(*args):
        lazy_input_a, lazy_input_b = convert_torch_tensors_to_lazy_tensors(*args, allow_scalars=True)
        lazy_output = lazy_input_b * lazy_input_a
        if run_torch:
            output = TORCH_TENSOR["__mul__"](*args)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def truediv(run_torch):
    def implementation(*args):
        lazy_input_a, lazy_input_b = convert_torch_tensors_to_lazy_tensors(*args, allow_scalars=True)
        lazy_output = lazy_input_a / lazy_input_b
        if run_torch:
            output = TORCH_TENSOR["__truediv__"](*args)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def neg(run_torch):
    def implementation(*args):
        (lazy_input,) = convert_torch_tensors_to_lazy_tensors(*args)
        lazy_output = lazy_input * -1
        if run_torch:
            output = TORCH["neg"](*args)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def getitem(run_torch):
    def implementation(*args):
        lazy_input, *_ = convert_torch_tensors_to_lazy_tensors(*args)

        fake_input = torch.rand(lazy_input.shape)
        fake_output = TORCH_TENSOR["__getitem__"](fake_input, *args[1:])

        if math.prod(fake_input.shape) == math.prod(fake_output.shape):
            lazy_output = cnp.reshape(lazy_input, tuple(fake_output.shape))
        else:
            _, indices = args
            if isinstance(indices, int):
                indices = [indices]
            elif isinstance(indices, slice):
                indices = [indices]
            else:

                def process_index(index):
                    if isinstance(index, torch.Tensor):
                        index = index.tolist()
                        assert len(index) == 1
                        return index[0]
                    else:
                        return index

                indices = [process_index(index) for index in indices]
            lazy_output = cnp.get_item(lazy_input, indices)

        if run_torch:
            output = TORCH_TENSOR["__getitem__"](*args)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def view(run_torch):
    def implementation(*args):
        lazy_input, *_ = convert_torch_tensors_to_lazy_tensors(*args)

        fake_input = torch.rand(lazy_input.shape)
        fake_output = TORCH_TENSOR["view"](fake_input, *args[1:])

        lazy_output = cnp.reshape(lazy_input, fake_output.shape)

        if run_torch:
            output = TORCH_TENSOR["view"](*args)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def reshape(run_torch):
    def implementation(*args):
        lazy_input, *_ = convert_torch_tensors_to_lazy_tensors(*args)

        fake_input = torch.rand(lazy_input.shape)
        fake_output = TORCH_TENSOR["reshape"](fake_input, *args[1:])

        lazy_output = cnp.reshape(lazy_input, fake_output.shape)

        if run_torch:
            output = TORCH_TENSOR["reshape"](*args)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def expand(run_torch):
    def implementation(*args):
        lazy_input, *_ = convert_torch_tensors_to_lazy_tensors(*args)

        if lazy_input.shape != (1,):
            logger.warning("Defaulting to torch.Tensor.expand")
            torch_input = torch.from_numpy(cnp.evaluate(lazy_input))
            output = TORCH_TENSOR["expand"](torch_input, *args[1:])
            return output

        lazy_output = cnp.concatenate([lazy_input, lazy_input], axis=0)

        if run_torch:
            output = TORCH_TENSOR["expand"](*args)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def contiguous(run_torch):
    def implementation(*args):
        lazy_input, *_ = convert_torch_tensors_to_lazy_tensors(*args)
        lazy_output = lazy_input
        if run_torch:
            output = TORCH_TENSOR["contiguous"](*args)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def to(run_torch):
    def implementation(*args, **kwargs):
        lazy_input, *_ = convert_torch_tensors_to_lazy_tensors(*args)
        if len(args) == 2:
            _, target = args
        else:
            assert len(args) == 1
            target = kwargs["dtype"]
        if isinstance(target, torch.dtype):
            numpy_dtype = {torch.float32: np.float32, torch.bool: bool, torch.int32: np.int32, torch.int64: np.int64}[
                target
            ]
            lazy_output = cnp.astype(lazy_input, dtype=numpy_dtype)
        else:
            logger.warning(f"to({target}) is not implemented")
            lazy_output = lazy_input

        if run_torch:
            output = TORCH_TENSOR["to"](*args, **kwargs)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def float(run_torch):
    def implementation(*args):
        lazy_input, *_ = convert_torch_tensors_to_lazy_tensors(*args)
        lazy_output = lazy_input
        if run_torch:
            output = TORCH_TENSOR["float"](*args)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def permute(run_torch):
    def implementation(*args):
        lazy_input, *_ = convert_torch_tensors_to_lazy_tensors(*args)
        order = args[1:]
        if len(order) == 1 and isinstance(order[0], (tuple, list)):
            order = tuple(order[0])
        lazy_output = cnp.transpose(lazy_input, order)
        if run_torch:
            output = TORCH_TENSOR["permute"](*args)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def transpose(run_torch):
    def implementation(*args):
        lazy_input, *_ = convert_torch_tensors_to_lazy_tensors(*args)

        rank = len(lazy_input.shape)
        axes_to_transpose = list(args[1:])
        assert len(axes_to_transpose) == 2
        for index, axis in enumerate(axes_to_transpose):
            axes_to_transpose[index] = (rank + axis) % rank

        if axes_to_transpose[0] < axes_to_transpose[1]:
            axes_to_transpose = tuple(reversed(axes_to_transpose))

        order = []
        axes_to_transpose_index = 0
        for axis in range(rank):
            if axis not in axes_to_transpose:
                order.append(axis)
            else:
                order.append(axes_to_transpose[axes_to_transpose_index])
                axes_to_transpose_index += 1
        order = tuple(order)

        lazy_output = cnp.transpose(lazy_input, order)
        if run_torch:
            output = TORCH_TENSOR["transpose"](*args)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def chunk(run_torch):
    def implementation(*args, **kwargs):
        lazy_input, *_ = convert_torch_tensors_to_lazy_tensors(*args)
        _, num_chunks = args
        axis = kwargs["dim"]
        lazy_outputs = cnp.split(lazy_input, num_chunks, axis=axis)
        if run_torch:
            outputs = TORCH_TENSOR["chunk"](*args, **kwargs)
            return tuple(Tensor(output, lazy_output) for output, lazy_output in zip(outputs, lazy_outputs))
        else:
            return lazy_outputs

    return implementation


def masked_fill(run_torch):
    def implementation(*args, **kwargs):
        @cnp.wrap_as_operation()
        def masked_fill(input_tensor, mask, value):
            input_tensor = torch.from_numpy(input_tensor)
            mask = torch.from_numpy(mask)
            value = torch.from_numpy(value)
            output_tensor = TORCH_TENSOR["masked_fill"](input_tensor, mask, value)
            output_tensor = output_tensor.detach().numpy()
            return output_tensor

        lazy_input, lazy_mask = convert_torch_tensors_to_lazy_tensors(*args)
        value = args[2]
        lazy_output = masked_fill(lazy_input, lazy_mask, value)

        if run_torch:
            output = TORCH_TENSOR["masked_fill"](*args, **kwargs)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def matmul(run_torch):
    def implementation(*args):
        lazy_input_a, lazy_input_b = convert_torch_tensors_to_lazy_tensors(*args)
        lazy_output = lazy_input_a @ lazy_input_b
        if run_torch:
            output = TORCH["matmul"](*args)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def bmm(run_torch):
    def implementation(*args):
        lazy_input_a, lazy_input_b = convert_torch_tensors_to_lazy_tensors(*args)
        lazy_output = lazy_input_a @ lazy_input_b
        if run_torch:
            output = TORCH["bmm"](*args)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def abs(run_torch):
    def implementation(*args):
        (lazy_input,) = convert_torch_tensors_to_lazy_tensors(*args)
        lazy_output = cnp.abs(lazy_input)
        if run_torch:
            output = TORCH["abs"](*args)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def exp(run_torch):
    def implementation(*args):
        (lazy_input,) = convert_torch_tensors_to_lazy_tensors(*args)
        lazy_output = cnp.exp(lazy_input)
        if run_torch:
            output = TORCH["exp"](*args)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def sin(run_torch):
    def implementation(*args):
        (lazy_input,) = convert_torch_tensors_to_lazy_tensors(*args)
        lazy_output = cnp.sin(lazy_input)
        if run_torch:
            output = TORCH["sin"](*args)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def cos(run_torch):
    def implementation(*args):
        (lazy_input,) = convert_torch_tensors_to_lazy_tensors(*args)
        lazy_output = cnp.cos(lazy_input)
        if run_torch:
            output = TORCH["cos"](*args)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def tanh(run_torch):
    def implementation(*args):
        (lazy_input,) = convert_torch_tensors_to_lazy_tensors(*args)
        lazy_output = cnp.tanh(lazy_input)
        if run_torch:
            output = TORCH["tanh"](*args)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def rsqrt(run_torch):
    def implementation(*args):
        (lazy_input,) = convert_torch_tensors_to_lazy_tensors(*args)
        lazy_output = cnp.reciprocal(cnp.sqrt(lazy_input))
        if run_torch:
            output = TORCH["rsqrt"](*args)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def sigmoid(run_torch):
    def implementation(*args):
        (lazy_input,) = convert_torch_tensors_to_lazy_tensors(*args)
        lazy_output = cnp.nn.sigmoid(lazy_input)
        if run_torch:
            output = TORCH["sigmoid"](*args)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def pow(run_torch):
    def implementation(*args):
        lazy_input_a, power = convert_torch_tensors_to_lazy_tensors(*args, allow_scalars=True)
        lazy_output = cnp.power(lazy_input_a, power)
        if run_torch:
            output = TORCH["pow"](*args)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def log(run_torch):
    def implementation(*args):
        (lazy_input,) = convert_torch_tensors_to_lazy_tensors(*args)
        lazy_output = cnp.log(lazy_input)
        if run_torch:
            output = TORCH["log"](*args)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def mean(run_torch):
    def implementation(*args, **kwargs):
        assert len(kwargs) == 1 and "keepdim" in kwargs
        (lazy_input,) = convert_torch_tensors_to_lazy_tensors(*args)
        _, axis = args
        keepdims = kwargs["keepdim"]
        lazy_output = cnp.mean(lazy_input, axis, keepdims=keepdims)
        if run_torch:
            output = TORCH["mean"](*args, **kwargs)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def max(run_torch):  # TODO: implement properly
    def implementation(*args, **kwargs):
        if not run_torch:
            raise RuntimeError("This can only be executed when running with torch")
        if run_torch:
            output = TORCH["max"](*args, **kwargs)
            return Tensor(output, cnp.asarray(output.detach().numpy()))
        else:
            ...

    return implementation


def min(run_torch):  # TODO: implement properly
    def implementation(*args, **kwargs):
        if not run_torch:
            raise RuntimeError("This can only be executed when running with torch")
        if run_torch:
            output = TORCH["min"](*args, **kwargs)
            return Tensor(output, cnp.asarray(output.detach().numpy()))
        else:
            ...

    return implementation


def cat(run_torch):
    def implementation(*args, **kwargs):
        input_tensors, *_ = args
        lazy_inputs = list(convert_torch_tensors_to_lazy_tensors(*input_tensors))

        axis = kwargs.get("dim", kwargs.get("axis", args[1] if len(args) > 1 else 0))
        lazy_output = cnp.concatenate(lazy_inputs, axis)
        if run_torch:
            output = TORCH["cat"](*args, **kwargs)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def flatten(run_torch):
    def implementation(*args, **kwargs):
        assert len(kwargs) == 0
        lazy_input, *_ = convert_torch_tensors_to_lazy_tensors(*args)
        lazy_output = cnp.reshape(lazy_input, (math.prod(lazy_input.shape),))
        if run_torch:
            output = TORCH["flatten"](*args, **kwargs)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def squeeze(run_torch):
    def implementation(*args, **kwargs):
        assert len(kwargs) == 0
        lazy_input, *_ = convert_torch_tensors_to_lazy_tensors(*args)
        _, axis = args
        old_shape = lazy_input.shape
        axis = (axis + len(old_shape)) % len(old_shape)
        new_shape = tuple(old_shape[:axis]) + tuple(old_shape[axis + 1 :])
        lazy_output = cnp.reshape(lazy_input, new_shape)
        if run_torch:
            output = TORCH["squeeze"](*args, **kwargs)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def unsqueeze(run_torch):
    def implementation(*args, **kwargs):
        assert len(kwargs) == 0
        lazy_input, *_ = convert_torch_tensors_to_lazy_tensors(*args)
        _, axis = args
        old_shape = lazy_input.shape
        axis = (axis + len(old_shape)) % len(old_shape)
        new_shape = tuple(old_shape[:axis]) + (1,) + tuple(old_shape[axis:])
        lazy_output = cnp.reshape(lazy_input, new_shape)
        if run_torch:
            output = TORCH["unsqueeze"](*args, **kwargs)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def linear(run_torch):
    def implementation(*args, **kwargs):
        assert len(kwargs) == 0
        lazy_input, lazy_weight, *rest = convert_torch_tensors_to_lazy_tensors(*args)
        lazy_output = lazy_input @ cnp.transpose(lazy_weight, (1, 0))
        if len(rest) == 1:
            (lazy_bias,) = rest
            lazy_output += lazy_bias

        if run_torch:
            output = TORCH_NN_FUNCTIONAL["linear"](*args, **kwargs)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def conv1d(run_torch):
    def implementation(*args, **kwargs):
        assert len(kwargs) == 0

        lazy_input, lazy_weight, *rest = convert_torch_tensors_to_lazy_tensors(*args)

        strides, padding, dilation, groups = args[3:]
        assert (
            isinstance(padding, tuple) and len(padding) == 1 and all(isinstance(element, int) for element in padding)
        ), "padding should be a tuple of 1 integers"
        assert dilation == (1,), f"dilation should be (1,) but is {dilation}"
        assert groups == 1, f"groups should be 1 but is {groups}"

        lazy_input = cnp.transpose(lazy_input, (0, 2, 1))
        lazy_weight = cnp.transpose(lazy_weight, (0, 2, 1))
        lazy_output = cnp.nn.convolution(lazy_input, lazy_weight, strides=strides, padding=padding, channels_last=True)

        if len(rest) == 1:
            (lazy_bias,) = rest
            lazy_output += lazy_bias

        lazy_output = cnp.transpose(lazy_output, (0, 2, 1))

        if run_torch:
            output = TORCH_NN_FUNCTIONAL["conv1d"](*args, **kwargs)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def conv2d(run_torch):
    def implementation(*args, **kwargs):
        assert len(kwargs) == 0

        lazy_input, lazy_weight, *rest = convert_torch_tensors_to_lazy_tensors(*args)

        strides, padding, dilation, groups = args[3:]
        assert (
            isinstance(padding, tuple) and len(padding) == 2 and all(isinstance(element, int) for element in padding)
        ), "padding should be a tuple of 2 integers"
        assert dilation == (1, 1), f"dilation should be (1, 1) but is {dilation}"
        assert groups == 1, f"groups should be 1 but is {groups}"

        lazy_input = cnp.transpose(lazy_input, (0, 2, 3, 1))
        lazy_weight = cnp.transpose(lazy_weight, (0, 2, 3, 1))
        lazy_output = cnp.nn.convolution(lazy_input, lazy_weight, strides=strides, padding=padding, channels_last=True)

        if len(rest) == 1:
            (lazy_bias,) = rest
            lazy_output += lazy_bias

        lazy_output = cnp.transpose(lazy_output, (0, 3, 1, 2))

        if run_torch:
            output = TORCH_NN_FUNCTIONAL["conv2d"](*args, **kwargs)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def max_pool2d(run_torch):
    def implementation(*args, **kwargs):
        (lazy_input,) = convert_torch_tensors_to_lazy_tensors(*args)

        if len(args) == 5:
            kernel_size, strides, padding, dilation = args[1:]
        else:
            (kernel_size,) = args[1:]
            strides = kwargs["stride"]
            padding = kwargs["padding"]
            dilation = kwargs["dilation"]

        assert isinstance(kernel_size, int), "kernel_size should be an integer"
        assert isinstance(strides, int), "strides should be an integer"
        assert isinstance(padding, int), "padding should be an integer"
        assert isinstance(dilation, int), "dilation should be an integer"

        kernel_size = (kernel_size, kernel_size)
        strides = (strides, strides)
        padding = (padding, padding)

        lazy_input = cnp.transpose(lazy_input, (0, 2, 3, 1))
        lazy_output = cnp.nn.max_pool(
            lazy_input, kernel_size=kernel_size, strides=strides, padding=padding, channels_last=True
        )
        lazy_output = cnp.transpose(lazy_output, (0, 3, 1, 2))

        if run_torch:
            output = TORCH_NN_FUNCTIONAL["max_pool2d"](*args, **kwargs)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def adaptive_avg_pool2d(run_torch):
    def implementation(*args, **kwargs):
        if len(args) == 2:
            (output_size,) = args[1:]
        else:
            output_size = kwargs["output_size"]
        assert output_size == (1, 1)

        (lazy_input,) = convert_torch_tensors_to_lazy_tensors(*args)
        lazy_output = cnp.mean(lazy_input, axis=(2, 3))

        if run_torch:
            output = TORCH_NN_FUNCTIONAL["adaptive_avg_pool2d"](*args, **kwargs)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def embedding(run_torch):
    def implementation(*args, **kwargs):
        input_ids, weights = convert_torch_tensors_to_lazy_tensors(*args)
        lazy_output = cnp.nn.embedding(input_ids, weights)

        if run_torch:
            output = TORCH_NN_FUNCTIONAL["embedding"](*args, **kwargs)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def relu(run_torch):
    def implementation(*args, **kwargs):
        (lazy_input,) = convert_torch_tensors_to_lazy_tensors(*args)
        lazy_output = cnp.nn.relu(lazy_input)
        if run_torch:
            output = TORCH_NN_FUNCTIONAL["relu"](*args, **kwargs)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def gelu(run_torch):
    def implementation(*args, **kwargs):
        assert len(kwargs) == 0
        (lazy_input,) = convert_torch_tensors_to_lazy_tensors(*args)
        lazy_output = cnp.nn.gelu(lazy_input)
        if run_torch:
            output = TORCH_NN_FUNCTIONAL["gelu"](*args, **kwargs)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def silu(run_torch):
    def implementation(*args, **kwargs):
        (lazy_input,) = convert_torch_tensors_to_lazy_tensors(*args)
        lazy_output = cnp.nn.silu(lazy_input)
        if run_torch:
            output = TORCH_NN_FUNCTIONAL["silu"](*args, **kwargs)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def softmax(run_torch):
    def implementation(*args, **kwargs):
        _, *rest = args
        if len(rest) > 0:
            axis, *_ = rest
        else:
            axis = kwargs["dim"]

        (lazy_input,) = convert_torch_tensors_to_lazy_tensors(*args)
        lazy_output = cnp.nn.layers.softmax(lazy_input, axis=axis)

        if run_torch:
            output = TORCH_NN_FUNCTIONAL["softmax"](*args, **kwargs)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def batch_norm(run_torch):
    def implementation(*args, **kwargs):
        lazy_inputs = convert_torch_tensors_to_lazy_tensors(*args)

        lazy_input, lazy_running_mean, lazy_running_var, *rest = lazy_inputs
        if len(rest) == 2:
            lazy_weight, lazy_bias = rest
        else:
            assert len(rest) == 0
            lazy_weight, lazy_bias = convert_torch_tensors_to_lazy_tensors(kwargs["weight"], kwargs["bias"])

        lazy_output = cnp.nn.layers.batch_norm(
            lazy_input,
            cnp.reshape(lazy_running_mean, (-1, 1, 1)),
            cnp.reshape(lazy_running_var, (-1, 1, 1)),
            cnp.reshape(lazy_weight, (-1, 1, 1)),
            cnp.reshape(lazy_bias, (-1, 1, 1)),
        )

        if run_torch:
            output = TORCH_NN_FUNCTIONAL["batch_norm"](*args, **kwargs)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def layer_norm(run_torch):
    def implementation(*args, **kwargs):
        lazy_inputs = convert_torch_tensors_to_lazy_tensors(*args)
        lazy_input, *rest = lazy_inputs
        if len(rest) == 2:
            lazy_weight, lazy_bias = rest
        else:
            assert len(rest) == 0
            lazy_weight, lazy_bias = convert_torch_tensors_to_lazy_tensors(kwargs["weight"], kwargs["bias"])

        lazy_output = cnp.nn.layers.layer_norm(lazy_input, lazy_weight, lazy_bias)

        if run_torch:
            output = TORCH_NN_FUNCTIONAL["layer_norm"](*args, **kwargs)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def group_norm(run_torch):
    def implementation(*args, **kwargs):
        lazy_inputs = convert_torch_tensors_to_lazy_tensors(*args)
        lazy_input, *rest = lazy_inputs
        if len(rest) == 2:
            lazy_weight, lazy_bias = rest
        else:
            assert len(rest) == 0
            lazy_weight, lazy_bias = convert_torch_tensors_to_lazy_tensors(kwargs["weight"], kwargs["bias"])

        num_groups = args[1]
        epsilon = args[4] if len(args) >= 5 else 1e-5

        lazy_output = cnp.nn.layers.group_norm(
            lazy_input,
            cnp.reshape(lazy_weight, (-1, 1, 1)),
            cnp.reshape(lazy_bias, (-1, 1, 1)),
            channel_axis=1,
            num_groups=num_groups,
            epsilon=epsilon,
        )

        if run_torch:
            output = TORCH_NN_FUNCTIONAL["group_norm"](*args, **kwargs)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def scaled_dot_product_attention(run_torch):
    def implementation(*args, **kwargs):
        lazy_query, lazy_key, lazy_value = convert_torch_tensors_to_lazy_tensors(*args)
        lazy_output = cnp.nn.layers.scaled_dot_product_attention(lazy_query, lazy_key, lazy_value)

        if run_torch:
            output = TORCH_NN_FUNCTIONAL["scaled_dot_product_attention"](*args, **kwargs)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


def interpolate(run_torch):
    def implementation(*args, **kwargs):
        @cnp.wrap_as_operation()
        def interpolate(input_tensor, *args, **kwargs):
            input_tensor = torch.from_numpy(input_tensor)
            output_tensor = TORCH_NN_FUNCTIONAL["interpolate"](input_tensor, *args, **kwargs)
            output_tensor = output_tensor.detach().numpy()
            return output_tensor

        (lazy_input,) = convert_torch_tensors_to_lazy_tensors(*args)
        lazy_output = interpolate(lazy_input, **kwargs)

        if run_torch:
            output = TORCH_NN_FUNCTIONAL["interpolate"](*args, **kwargs)
            return Tensor(output, lazy_output)
        else:
            return lazy_output

    return implementation


@contextmanager
def trace(*, run_torch=False):
    reset_graph_input_index()

    torch.__dict__.clear()
    for attr in TORCH_ATTRIBUTES_TO_LEAVE_AS_IS:
        setattr(torch, attr, TORCH[attr])

    torch.nn.functional.__dict__.clear()
    for attr in TORCH_NN_FUNCTIONAL_ATTRIBUTES_TO_LEAVE_AS_IS:
        setattr(torch.nn.functional, attr, TORCH_NN_FUNCTIONAL[attr])

    for attr in TORCH_TENSOR:
        if attr in TORCH_TENSOR_ATTRIBUTES_TO_LEAVE_AS_IS:
            continue

        def not_implemented(attr):
            def raise_exception(*args, **kwargs):
                raise NotImplementedError(f"{attr} is not implemented")

            return raise_exception

        setattr(torch.Tensor, attr, not_implemented(attr))

    # Overrides

    setattr(torch, "matmul", matmul(run_torch))
    setattr(torch, "bmm", bmm(run_torch))
    setattr(torch, "neg", neg(run_torch))
    setattr(torch, "abs", abs(run_torch))
    setattr(torch, "exp", exp(run_torch))
    setattr(torch, "sin", sin(run_torch))
    setattr(torch, "cos", cos(run_torch))
    setattr(torch, "tanh", tanh(run_torch))
    setattr(torch, "rsqrt", rsqrt(run_torch))
    setattr(torch, "sigmoid", sigmoid(run_torch))
    setattr(torch, "pow", pow(run_torch))
    setattr(torch, "log", log(run_torch))
    setattr(torch, "cat", cat(run_torch))
    setattr(torch, "flatten", flatten(run_torch))
    setattr(torch, "mean", mean(run_torch))
    setattr(torch, "max", max(run_torch))
    setattr(torch, "min", min(run_torch))

    setattr(torch.nn.functional, "linear", linear(run_torch))
    setattr(torch.nn.functional, "conv1d", conv1d(run_torch))
    setattr(torch.nn.functional, "conv2d", conv2d(run_torch))
    setattr(torch.nn.functional, "max_pool2d", max_pool2d(run_torch))
    setattr(torch.nn.functional, "adaptive_avg_pool2d", adaptive_avg_pool2d(run_torch))
    setattr(torch.nn.functional, "embedding", embedding(run_torch))
    setattr(torch.nn.functional, "dropout", identity(run_torch))
    setattr(torch.nn.functional, "softmax", softmax(run_torch))
    setattr(torch.nn.functional, "relu", relu(run_torch))
    setattr(torch.nn.functional, "gelu", gelu(run_torch))
    setattr(torch.nn.functional, "silu", silu(run_torch))
    setattr(torch.nn.functional, "mish", identity(run_torch))
    setattr(torch.nn.functional, "batch_norm", batch_norm(run_torch))
    setattr(torch.nn.functional, "layer_norm", layer_norm(run_torch))
    setattr(torch.nn.functional, "group_norm", group_norm(run_torch))
    setattr(torch.nn.functional, "scaled_dot_product_attention", scaled_dot_product_attention(run_torch))
    setattr(torch.nn.functional, "interpolate", interpolate(run_torch))

    setattr(torch.Tensor, "__add__", add(run_torch))
    setattr(torch.Tensor, "__iadd__", add(run_torch))
    setattr(torch.Tensor, "__radd__", radd(run_torch))
    setattr(torch.Tensor, "__sub__", sub(run_torch))
    setattr(torch.Tensor, "__rsub__", rsub(run_torch))
    setattr(torch.Tensor, "__mul__", mul(run_torch))
    setattr(torch.Tensor, "__rmul__", rmul(run_torch))
    setattr(torch.Tensor, "__truediv__", truediv(run_torch))
    setattr(torch.Tensor, "__neg__", neg(run_torch))
    setattr(torch.Tensor, "__matmul__", matmul(run_torch))
    setattr(torch.Tensor, "__getitem__", getitem(run_torch))
    setattr(torch.Tensor, "view", view(run_torch))
    setattr(torch.Tensor, "reshape", reshape(run_torch))
    setattr(torch.Tensor, "expand", expand(run_torch))
    setattr(torch.Tensor, "unsqueeze", unsqueeze(run_torch))
    setattr(torch.Tensor, "squeeze", squeeze(run_torch))
    setattr(torch.Tensor, "permute", permute(run_torch))
    setattr(torch.Tensor, "transpose", transpose(run_torch))
    setattr(torch.Tensor, "contiguous", contiguous(run_torch))
    setattr(torch.Tensor, "to", to(run_torch))
    setattr(torch.Tensor, "float", float(run_torch))
    setattr(torch.Tensor, "chunk", chunk(run_torch))
    setattr(torch.Tensor, "masked_fill", masked_fill(run_torch))
    setattr(torch.Tensor, "pow", pow(run_torch))
    setattr(torch.Tensor, "mean", mean(run_torch))
    setattr(torch.Tensor, "max", max(run_torch))

    yield

    for attr, value in TORCH.items():
        setattr(torch, attr, value)

    for attr, value in TORCH_NN_FUNCTIONAL.items():
        setattr(torch.nn.functional, attr, value)

    for attr, value in TORCH_TENSOR.items():
        if attr in TORCH_TENSOR_ATTRIBUTES_TO_LEAVE_AS_IS:
            continue
        setattr(torch.Tensor, attr, value)
