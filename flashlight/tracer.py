from contextlib import contextmanager
import math

import torch
import torch.mps
import torch.nn.functional

import composit as cnp
import composit.nn
from flashlight.introspection import convert_torch_tensors_to_lazy_tensors
from flashlight import Tensor

TORCH = torch.__dict__.copy()
TORCH_TENSOR = {attr: getattr(torch.Tensor, attr) for attr in dir(torch.Tensor)}
TORCH_NN_FUNCTIONAL = torch.nn.functional.__dict__.copy()

TORCH_ATTRIBUTES_TO_LEAVE_AS_IS = [
    "arange",
    "autograd",
    "backends",
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
    "embedding",
    "embedding_renorm_",
    "empty",
    "finfo",
    "float",
    "float16",
    "float32",
    "FloatTensor",
    "_from_functional_tensor",
    "fx",
    "Generator",
    "get_default_dtype",
    "int",
    "int64",
    "isfinite",
    "_is_functional_tensor",
    "is_grad_enabled",
    "is_tensor",
    "jit",
    "_jit_internal",
    "layer_norm",
    "long",
    "LongTensor",
    "masked_select",
    "mps",
    "nn",
    "no_grad",
    "optim",
    "overrides",
    "rand",
    "randn",
    "save",
    "set_grad_enabled",
    "Size",
    "sparse_csr",
    "sparse_csc",
    "sparse_bsr",
    "sparse_bsc",
    "strided",
    "SymFloat",
    "tensor",
    "Tensor",
    "Size",
    "_tensor_str",
    "torch",
    "utils",
    "__version__",
    "zeros",
]

TORCH_TENSOR_ATTRIBUTES_TO_LEAVE_AS_IS = {
    "__class__",
    "__dict__",
    "__getattribute__",
    "__hash__",
    "__init__",
    "__new__",
    "__repr__",
    "__setattr__",
    "__torch_function__",
    "__torch_dispatch__",
    "_make_subclass",
    "add",  # TODO: override?
    "argmax",  # TODO: override?
    "as_subclass",
    "detach",  # TODO: potentially override?
    "dim",
    "div",  # TODO: override?
    "fill_",  # TODO: override?
    "expand",  # TODO: override with reshape
    "is_floating_point",
    "matmul",  # TODO: override?
    "mul",  # TODO: override?
    "normal_",  # TODO: override?
    "numel",
    "numpy",
    "new",
    "size",
    "softmax",  # TODO: override?
    "sub",  # TODO: override?
    "tolist",
    "truediv",  # TODO: override?
    "uniform_",  # TODO: override?
    "zero_",  # TODO: override?
}

TORCH_NN_FUNCTIONAL_ATTRIBUTES_TO_LEAVE_AS_IS = [
    "handle_torch_function",
    "has_torch_function_unary",
    "has_torch_function_variadic",
    "_no_grad_embedding_renorm_",
    "torch",
    "_verify_batch_size",
]


def add(*args):
    output = TORCH_TENSOR["__add__"](*args)
    lazy_input_a, lazy_input_b = convert_torch_tensors_to_lazy_tensors(*args, allow_scalars=True)
    lazy_output = lazy_input_a + lazy_input_b
    return Tensor(output, lazy_output)


def sub(*args):
    output = TORCH_TENSOR["__sub__"](*args)
    lazy_input_a, lazy_input_b = convert_torch_tensors_to_lazy_tensors(*args, allow_scalars=True)
    lazy_output = lazy_input_a + lazy_input_b
    return Tensor(output, lazy_output)


def rsub(*args):
    output = TORCH_TENSOR["__sub__"](*args)
    lazy_input_b, lazy_input_a = convert_torch_tensors_to_lazy_tensors(*args, allow_scalars=True)
    lazy_output = lazy_input_a - lazy_input_b
    return Tensor(output, lazy_output)


def mul(*args):
    output = TORCH_TENSOR["__mul__"](*args)
    lazy_input_a, lazy_input_b = convert_torch_tensors_to_lazy_tensors(*args, allow_scalars=True)
    lazy_output = lazy_input_a * lazy_input_b
    return Tensor(output, lazy_output)


def rmul(*args):
    output = TORCH_TENSOR["__rmul__"](*args)
    lazy_input_b, lazy_input_a = convert_torch_tensors_to_lazy_tensors(*args, allow_scalars=True)
    lazy_output = lazy_input_a * lazy_input_b
    return Tensor(output, lazy_output)


def truediv(*args):
    output = TORCH_TENSOR["__truediv__"](*args)
    lazy_input_a, lazy_input_b = convert_torch_tensors_to_lazy_tensors(*args, allow_scalars=True)
    lazy_output = lazy_input_a / lazy_input_b
    return Tensor(output, lazy_output)


def matmul(*args):
    output = TORCH_TENSOR["__matmul__"](*args)
    lazy_input_a, lazy_input_b = convert_torch_tensors_to_lazy_tensors(*args)
    lazy_output = lazy_input_a @ lazy_input_b
    return Tensor(output, lazy_output)


def getitem(*args):
    output = TORCH_TENSOR["__getitem__"](*args)
    lazy_input, *_ = convert_torch_tensors_to_lazy_tensors(*args)
    if math.prod(output.shape) == math.prod(lazy_input.shape):
        lazy_output = cnp.reshape(lazy_input, tuple(output.shape))
    else:
        _, indices = args
        if isinstance(indices, int):
            indices = [indices]
        else:
            indices = [index.tolist() if isinstance(index, torch.Tensor) else index for index in indices]
        lazy_output = cnp.get_item(lazy_input, indices)
    return Tensor(output, lazy_output)


def view(*args):
    output = TORCH_TENSOR["view"](*args)
    lazy_input, *_ = convert_torch_tensors_to_lazy_tensors(*args)
    return Tensor(output, cnp.reshape(lazy_input, tuple(output.shape)))


def reshape(*args):
    output = TORCH_TENSOR["reshape"](*args)
    lazy_input, *_ = convert_torch_tensors_to_lazy_tensors(*args)
    return Tensor(output, cnp.reshape(lazy_input, tuple(output.shape)))


def contiguous(*args):
    output = TORCH_TENSOR["contiguous"](*args)
    lazy_input, *_ = convert_torch_tensors_to_lazy_tensors(*args)
    return Tensor(output, lazy_input)


def to(*args, **kwargs):
    output = TORCH_TENSOR["to"](*args, **kwargs)
    lazy_input, *_ = convert_torch_tensors_to_lazy_tensors(*args)
    return Tensor(output, lazy_input)


def float(*args, **kwargs):
    output = TORCH_TENSOR["float"](*args, **kwargs)
    lazy_input, *_ = convert_torch_tensors_to_lazy_tensors(*args)
    return Tensor(output, lazy_input)


def permute(*args):
    output = TORCH_TENSOR["permute"](*args)
    (lazy_input,) = convert_torch_tensors_to_lazy_tensors(*args)
    order = args[1:]
    return Tensor(output, cnp.transpose(lazy_input, order))


def transpose(*args):
    output = TORCH_TENSOR["transpose"](*args)
    (lazy_input,) = convert_torch_tensors_to_lazy_tensors(*args)

    axes_to_transpose = list(args[1:])
    assert len(axes_to_transpose) == 2
    for index, axis in enumerate(axes_to_transpose):
        axes_to_transpose[index] = (len(output.shape) + axis) % len(output.shape)

    if axes_to_transpose[0] < axes_to_transpose[1]:
        axes_to_transpose = tuple(reversed(axes_to_transpose))

    order = []
    axes_to_transpose_index = 0
    for axis, _ in enumerate(output.shape):
        if axis not in axes_to_transpose:
            order.append(axis)
        else:
            order.append(axes_to_transpose[axes_to_transpose_index])
            axes_to_transpose_index += 1
    order = tuple(order)

    return Tensor(output, cnp.transpose(lazy_input, order))


def identity(*args, **kwargs):
    output, *_ = args
    (lazy_input,) = convert_torch_tensors_to_lazy_tensors(*args)
    return Tensor(output, lazy_input)


def bmm(*args):
    output = TORCH["bmm"](*args)
    lazy_input_a, lazy_input_b = convert_torch_tensors_to_lazy_tensors(*args)
    lazy_output = lazy_input_a @ lazy_input_b
    return Tensor(output, lazy_output)


def exp(*args):
    output = TORCH["exp"](*args)
    (lazy_input,) = convert_torch_tensors_to_lazy_tensors(*args)
    lazy_output = cnp.exp(lazy_input)
    return Tensor(output, lazy_output)


def sin(*args):
    output = TORCH["sin"](*args)
    (lazy_input,) = convert_torch_tensors_to_lazy_tensors(*args)
    lazy_output = cnp.sin(lazy_input)
    return Tensor(output, lazy_output)


def cos(*args):
    output = TORCH["cos"](*args)
    (lazy_input,) = convert_torch_tensors_to_lazy_tensors(*args)
    lazy_output = cnp.cos(lazy_input)
    return Tensor(output, lazy_output)


def tanh(*args):
    output = TORCH["tanh"](*args)
    (lazy_input,) = convert_torch_tensors_to_lazy_tensors(*args)
    lazy_output = cnp.tanh(lazy_input)
    return Tensor(output, lazy_output)


def sigmoid(*args):
    output = TORCH["sigmoid"](*args)
    (lazy_input,) = convert_torch_tensors_to_lazy_tensors(*args)
    lazy_output = cnp.nn.sigmoid(lazy_input)
    return Tensor(output, lazy_output)


def cat(*args, **kwargs):
    output = TORCH["cat"](*args, **kwargs)

    input_tensors, *_ = args
    lazy_inputs = list(convert_torch_tensors_to_lazy_tensors(*input_tensors))

    axis = kwargs["dim"]
    lazy_output = cnp.concatenate(lazy_inputs, axis)
    return Tensor(output, lazy_output)


def linear(*args, **kwargs):
    output = TORCH_NN_FUNCTIONAL["linear"](*args, **kwargs)

    lazy_input, lazy_weight, *rest = convert_torch_tensors_to_lazy_tensors(*args)
    lazy_output = lazy_input @ cnp.transpose(lazy_weight, (1, 0))
    if len(rest) == 1:
        (lazy_bias,) = rest
        lazy_output += lazy_bias

    return Tensor(output, lazy_output)


def conv2d(*args, **kwargs):
    output = TORCH_NN_FUNCTIONAL["conv2d"](*args, **kwargs)

    lazy_input, lazy_weight, *rest = convert_torch_tensors_to_lazy_tensors(*args)
    lazy_output = cnp.nn.convolution(lazy_input, lazy_weight, channels_last=False)
    if len(rest) == 1:
        (lazy_bias,) = rest
        lazy_output += lazy_bias

    return Tensor(output, lazy_output)


def embedding(*args, **kwargs):
    output = TORCH_NN_FUNCTIONAL["embedding"](*args, **kwargs)

    input_ids, weights = convert_torch_tensors_to_lazy_tensors(*args)
    lazy_output = cnp.nn.embedding(input_ids, weights)
    return Tensor(output, lazy_output)


def gelu(*args, **kwargs):
    output = TORCH_NN_FUNCTIONAL["gelu"](*args, **kwargs)

    (lazy_input,) = convert_torch_tensors_to_lazy_tensors(*args)
    lazy_output = cnp.nn.gelu(lazy_input)
    return Tensor(output, lazy_output)


def silu(*args, **kwargs):
    output = TORCH_NN_FUNCTIONAL["silu"](*args, **kwargs)

    (lazy_input,) = convert_torch_tensors_to_lazy_tensors(*args)
    lazy_output = cnp.nn.silu(lazy_input)
    return Tensor(output, lazy_output)


def softmax(*args, **kwargs):
    output = TORCH_NN_FUNCTIONAL["softmax"](*args, **kwargs)

    _, *rest = args
    if len(rest) > 0:
        axis, *_ = rest
    else:
        axis = kwargs["dim"]

    (lazy_input,) = convert_torch_tensors_to_lazy_tensors(*args)
    lazy_output = cnp.nn.layers.softmax(lazy_input, axis=axis)
    return Tensor(output, lazy_output)


def layer_norm(*args, **kwargs):
    output = TORCH_NN_FUNCTIONAL["layer_norm"](*args, **kwargs)

    lazy_inputs = convert_torch_tensors_to_lazy_tensors(*args)
    lazy_input, *rest = lazy_inputs
    if len(rest) == 2:
        lazy_weight, lazy_bias = rest
    else:
        assert len(rest) == 0
        lazy_weight, lazy_bias = convert_torch_tensors_to_lazy_tensors(kwargs["weight"], kwargs["bias"])

    lazy_output = cnp.nn.layers.layer_norm(lazy_input, lazy_weight, lazy_bias)

    return Tensor(output, lazy_output)


@contextmanager
def trace():
    torch.__dict__.clear()
    for attr in TORCH_ATTRIBUTES_TO_LEAVE_AS_IS:
        setattr(torch, attr, TORCH[attr])

    torch.nn.functional.__dict__.clear()
    for attr in TORCH_NN_FUNCTIONAL_ATTRIBUTES_TO_LEAVE_AS_IS:
        setattr(torch.nn.functional, attr, TORCH_NN_FUNCTIONAL[attr])

    for attr in TORCH_TENSOR:
        if attr in TORCH_TENSOR_ATTRIBUTES_TO_LEAVE_AS_IS or not callable(TORCH_TENSOR[attr]):
            continue

        def not_implemented(attr):
            def raise_exception(*args, **kwargs):
                raise NotImplementedError(f"{attr} is not implemented")

            return raise_exception

        setattr(torch.Tensor, attr, not_implemented(attr))

    # Overrides

    setattr(torch, "matmul", matmul)
    setattr(torch, "bmm", bmm)
    setattr(torch, "exp", exp)
    setattr(torch, "sin", sin)
    setattr(torch, "cos", cos)
    setattr(torch, "tanh", tanh)
    setattr(torch, "sigmoid", sigmoid)
    setattr(torch, "cat", cat)

    setattr(torch.nn.functional, "linear", linear)
    setattr(torch.nn.functional, "conv2d", conv2d)
    setattr(torch.nn.functional, "embedding", embedding)
    setattr(torch.nn.functional, "layer_norm", layer_norm)
    setattr(torch.nn.functional, "dropout", identity)
    setattr(torch.nn.functional, "softmax", softmax)
    setattr(torch.nn.functional, "gelu", gelu)
    setattr(torch.nn.functional, "silu", silu)
    setattr(torch.nn.functional, "mish", identity)

    setattr(torch.Tensor, "__add__", add)
    setattr(torch.Tensor, "__sub__", sub)
    setattr(torch.Tensor, "__rsub__", rsub)
    setattr(torch.Tensor, "__mul__", mul)
    setattr(torch.Tensor, "__rmul__", rmul)
    setattr(torch.Tensor, "__truediv__", truediv)
    setattr(torch.Tensor, "__matmul__", matmul)
    setattr(torch.Tensor, "__getitem__", getitem)
    setattr(torch.Tensor, "view", view)
    setattr(torch.Tensor, "reshape", reshape)
    setattr(torch.Tensor, "permute", permute)
    setattr(torch.Tensor, "transpose", transpose)
    setattr(torch.Tensor, "contiguous", contiguous)
    setattr(torch.Tensor, "to", to)
    setattr(torch.Tensor, "float", float)

    yield

    for attr, value in TORCH.items():
        setattr(torch, attr, value)

    for attr, value in TORCH_NN_FUNCTIONAL.items():
        setattr(torch.nn.functional, attr, value)

    for attr, value in TORCH_TENSOR.items():
        if attr in TORCH_TENSOR_ATTRIBUTES_TO_LEAVE_AS_IS or not callable(TORCH_TENSOR[attr]):
            continue
        setattr(torch.Tensor, attr, value)
