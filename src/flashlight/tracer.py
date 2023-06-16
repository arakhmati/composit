from contextlib import contextmanager
import math

import torch
import torch.mps
import torch.nn.functional
from loguru import logger

import composit as cnp
import composit.nn
from flashlight.introspection import convert_torch_tensors_to_lazy_tensors, reset_graph_input_index
from flashlight import Tensor

TORCH = torch.__dict__.copy()
TORCH_TENSOR = {attr: getattr(torch.Tensor, attr) for attr in dir(torch.Tensor)}
TORCH_NN_FUNCTIONAL = torch.nn.functional.__dict__.copy()

TORCH_ATTRIBUTES_TO_LEAVE_AS_IS = [
    "arange",
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
    "fx",
    "Generator",
    "get_default_dtype",
    "group_norm",  # TODO: override?
    "int",
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
    "mps",
    "nn",
    "no_grad",
    "optim",
    "overrides",
    "rand",
    "randn",
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
    "_list_with_default",
    "_no_grad_embedding_renorm_",
    "torch",
    "_verify_batch_size",
]


def identity(*args, **kwargs):
    output, *_ = args
    (lazy_input,) = convert_torch_tensors_to_lazy_tensors(*args)
    return Tensor(output, lazy_input)


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


def expand(*args):
    output = TORCH_TENSOR["expand"](*args)
    lazy_input, *_ = convert_torch_tensors_to_lazy_tensors(*args)
    if lazy_input.shape == (1,):
        return Tensor(output, cnp.concatenate([lazy_input, lazy_input], axis=0))
    else:
        logger.warning("Defaulting to torch.Tensor.expand")
        return output


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


def chunk(*args, **kwargs):
    outputs = TORCH_TENSOR["chunk"](*args, **kwargs)
    (lazy_input,) = convert_torch_tensors_to_lazy_tensors(*args)
    _, num_chunks = args
    axis = kwargs["dim"]
    lazy_outputs = cnp.split(lazy_input, num_chunks, axis=axis)
    return tuple(Tensor(output, lazy_output) for output, lazy_output in zip(outputs, lazy_outputs))


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


def flatten(*args):
    output = TORCH["flatten"](*args)
    lazy_input, *_ = convert_torch_tensors_to_lazy_tensors(*args)
    return Tensor(output, cnp.reshape(lazy_input, tuple(output.shape)))


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

    return Tensor(output, lazy_output)


def max_pool2d(*args, **kwargs):
    output = TORCH_NN_FUNCTIONAL["max_pool2d"](*args, **kwargs)

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

    return Tensor(output, lazy_output)


def adaptive_avg_pool2d(*args, **kwargs):
    output = TORCH_NN_FUNCTIONAL["adaptive_avg_pool2d"](*args, **kwargs)

    if len(args) == 2:
        (output_size,) = args[1:]
    else:
        output_size = kwargs["output_size"]
    assert output_size == (1, 1)

    (lazy_input,) = convert_torch_tensors_to_lazy_tensors(*args)
    lazy_output = cnp.mean(lazy_input, axis=(2, 3))
    return Tensor(output, lazy_output)


def embedding(*args, **kwargs):
    output = TORCH_NN_FUNCTIONAL["embedding"](*args, **kwargs)

    input_ids, weights = convert_torch_tensors_to_lazy_tensors(*args)
    lazy_output = cnp.nn.embedding(input_ids, weights)
    return Tensor(output, lazy_output)


def relu(*args, **kwargs):
    output = TORCH_NN_FUNCTIONAL["relu"](*args, **kwargs)

    (lazy_input,) = convert_torch_tensors_to_lazy_tensors(*args)
    lazy_output = cnp.nn.relu(lazy_input)
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


def batch_norm(*args, **kwargs):
    output = TORCH_NN_FUNCTIONAL["batch_norm"](*args, **kwargs)

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


def group_norm(*args, **kwargs):
    output = TORCH_NN_FUNCTIONAL["group_norm"](*args, **kwargs)

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

    return Tensor(output, lazy_output)


def scaled_dot_product_attention(*args, **kwargs):
    output = TORCH_NN_FUNCTIONAL["scaled_dot_product_attention"](*args, **kwargs)
    lazy_query, lazy_key, lazy_value = convert_torch_tensors_to_lazy_tensors(*args)
    lazy_output = cnp.nn.layers.scaled_dot_product_attention(lazy_query, lazy_key, lazy_value)
    return Tensor(output, lazy_output)


def interpolate(*args, **kwargs):
    from composit.nn import wrap_as_instruction

    @wrap_as_instruction()
    def interpolate(input_tensor, *args, **kwargs):
        input_tensor = torch.from_numpy(input_tensor)
        output_tensor = TORCH_NN_FUNCTIONAL["interpolate"](input_tensor, *args, **kwargs)
        output_tensor = output_tensor.detach().numpy()
        return output_tensor

    output = TORCH_NN_FUNCTIONAL["interpolate"](*args, **kwargs)
    (lazy_input,) = convert_torch_tensors_to_lazy_tensors(*args)
    lazy_output = interpolate(lazy_input, **kwargs)
    return Tensor(output, lazy_output)


@contextmanager
def trace():
    reset_graph_input_index()

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
    setattr(torch, "flatten", flatten)

    setattr(torch.nn.functional, "linear", linear)
    setattr(torch.nn.functional, "conv2d", conv2d)
    setattr(torch.nn.functional, "max_pool2d", max_pool2d)
    setattr(torch.nn.functional, "adaptive_avg_pool2d", adaptive_avg_pool2d)
    setattr(torch.nn.functional, "embedding", embedding)
    setattr(torch.nn.functional, "dropout", identity)
    setattr(torch.nn.functional, "softmax", softmax)
    setattr(torch.nn.functional, "relu", relu)
    setattr(torch.nn.functional, "gelu", gelu)
    setattr(torch.nn.functional, "silu", silu)
    setattr(torch.nn.functional, "mish", identity)
    setattr(torch.nn.functional, "batch_norm", batch_norm)
    setattr(torch.nn.functional, "layer_norm", layer_norm)
    setattr(torch.nn.functional, "group_norm", group_norm)
    setattr(torch.nn.functional, "scaled_dot_product_attention", scaled_dot_product_attention)
    setattr(torch.nn.functional, "interpolate", interpolate)

    setattr(torch.Tensor, "__add__", add)
    setattr(torch.Tensor, "__iadd__", add)
    setattr(torch.Tensor, "__sub__", sub)
    setattr(torch.Tensor, "__rsub__", rsub)
    setattr(torch.Tensor, "__mul__", mul)
    setattr(torch.Tensor, "__rmul__", rmul)
    setattr(torch.Tensor, "__truediv__", truediv)
    setattr(torch.Tensor, "__matmul__", matmul)
    setattr(torch.Tensor, "__getitem__", getitem)
    setattr(torch.Tensor, "view", view)
    setattr(torch.Tensor, "reshape", reshape)
    setattr(torch.Tensor, "expand", expand)
    setattr(torch.Tensor, "permute", permute)
    setattr(torch.Tensor, "transpose", transpose)
    setattr(torch.Tensor, "contiguous", contiguous)
    setattr(torch.Tensor, "to", to)
    setattr(torch.Tensor, "float", float)
    setattr(torch.Tensor, "chunk", chunk)

    yield

    for attr, value in TORCH.items():
        setattr(torch, attr, value)

    for attr, value in TORCH_NN_FUNCTIONAL.items():
        setattr(torch.nn.functional, attr, value)

    for attr, value in TORCH_TENSOR.items():
        if attr in TORCH_TENSOR_ATTRIBUTES_TO_LEAVE_AS_IS or not callable(TORCH_TENSOR[attr]):
            continue
        setattr(torch.Tensor, attr, value)
