from contextlib import contextmanager

import torch
import torch.nn.functional

import composit as cnp
import composit.nn
from flashlight.introspection import convert_torch_tensors_to_lazy_tensors
from flashlight import Tensor

F = {attr : value for attr, value in torch.nn.functional.__dict__.items()}

torch_add = torch.Tensor.__add__


def add(*args):
    output = torch_add(*args)
    lazy_input_a, lazy_input_b = convert_torch_tensors_to_lazy_tensors(*args, allow_scalars=True)
    lazy_output = lazy_input_a + lazy_input_b
    return Tensor(output, lazy_output)


torch_sub = torch.Tensor.__sub__
def sub(*args):
    output = torch_sub(*args)
    lazy_input_a, lazy_input_b = convert_torch_tensors_to_lazy_tensors(*args, allow_scalars=True)
    lazy_output = lazy_input_a + lazy_input_b
    return Tensor(output, lazy_output)


torch_mul = torch.Tensor.__mul__
def mul(*args):
    output = torch_mul(*args)
    lazy_input_a, lazy_input_b = convert_torch_tensors_to_lazy_tensors(*args, allow_scalars=True)
    lazy_output = lazy_input_a * lazy_input_b
    return Tensor(output, lazy_output)


torch_truediv = torch.Tensor.__truediv__
def truediv(*args):
    output = torch_truediv(*args)
    lazy_input_a, lazy_input_b = convert_torch_tensors_to_lazy_tensors(*args, allow_scalars=True)
    lazy_output = lazy_input_a / lazy_input_b
    return Tensor(output, lazy_output)


torch_matmul = torch.Tensor.__matmul__
def matmul(*args):
    output = torch_matmul(*args)
    lazy_input_a, lazy_input_b = convert_torch_tensors_to_lazy_tensors(*args, allow_scalars=True)
    lazy_output = lazy_input_a @ lazy_input_b
    return Tensor(output, lazy_output)


# torch_getitem = torch.Tensor.__getitem__
# def getitem(*args):
#     assert len(args) == 2
#     args = convert_torch_tensors_to_lazy_tensors(*args)
#     output = torch_getitem(*args)
#     slices = args[1]
#     return Tensor(output, args[0].lazy_tensor[slices])


torch_view = torch.Tensor.view
def view(*args):
    output = torch_view(*args)
    lazy_input, *_ = convert_torch_tensors_to_lazy_tensors(*args)
    return Tensor(output, cnp.reshape(lazy_input, tuple(output.shape)))



torch_contiguous = torch.Tensor.contiguous
def contiguous(*args):
    output = torch_contiguous(*args)
    lazy_input, *_ = convert_torch_tensors_to_lazy_tensors(*args)
    return Tensor(output, lazy_input)



torch_to = torch.Tensor.to
def to(*args, **kwargs):
    output = torch_to(*args, **kwargs)
    lazy_input, *_ = convert_torch_tensors_to_lazy_tensors(*args)
    return Tensor(output, lazy_input)


torch_permute = torch.Tensor.permute
def permute(*args):
    output = torch_permute(*args)
    lazy_input, = convert_torch_tensors_to_lazy_tensors(*args)
    order = args[1:]
    return Tensor(output, cnp.transpose(lazy_input, order))


torch_transpose = torch.Tensor.transpose
def transpose(*args):
    output = torch_transpose(*args)
    lazy_input, = convert_torch_tensors_to_lazy_tensors(*args)

    axes_to_transpose = list(args[1:])
    for index, axis in enumerate(axes_to_transpose):
        axes_to_transpose[index] = (len(output.shape) + axis) % len(output.shape)

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
    lazy_input, = convert_torch_tensors_to_lazy_tensors(*args)
    return Tensor(output, lazy_input)


def linear(*args, **kwargs):
    output = F["linear"](*args, **kwargs)

    lazy_input, lazy_weight, *rest = convert_torch_tensors_to_lazy_tensors(*args)
    lazy_output = lazy_input @ cnp.transpose(lazy_weight, (1, 0))
    if len(rest) == 1:
        lazy_bias, = rest
        lazy_output += lazy_bias

    return Tensor(output, lazy_output)


def embedding(*args, **kwargs):
    output = F["embedding"](*args, **kwargs)

    input_ids, weights = convert_torch_tensors_to_lazy_tensors(*args)
    lazy_output = cnp.nn.embedding(input_ids, weights)
    return Tensor(output, lazy_output)


def gelu(*args, **kwargs):
    output = F["gelu"](*args, **kwargs)

    lazy_input, = convert_torch_tensors_to_lazy_tensors(*args)
    lazy_output = cnp.nn.gelu(lazy_input)
    return Tensor(output, lazy_output)



def softmax(*args, **kwargs):
    output = F["softmax"](*args, **kwargs)

    _, *rest = args
    if len(rest) > 0:
        axis, *_ = rest
    else:
        axis = kwargs["dim"]

    lazy_input, = convert_torch_tensors_to_lazy_tensors(*args)
    lazy_output = cnp.nn.layers.softmax(lazy_input, axis=axis)
    return Tensor(output, lazy_output)



def layer_norm(*args, **kwargs):
    output = F["layer_norm"](*args, **kwargs)

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
    cached_torch_dict = torch.__dict__.copy()
    cached_torch_nn_functional_dict = torch.nn.functional.__dict__.copy()
    torch.nn.functional.__dict__.clear()

    setattr(torch, "matmul", matmul)

    setattr(torch.nn.functional, "linear", linear)
    setattr(torch.nn.functional, "embedding", embedding)
    setattr(torch.nn.functional, "layer_norm", layer_norm)
    setattr(torch.nn.functional, "dropout", identity)
    setattr(torch.nn.functional, "softmax", softmax)
    setattr(torch.nn.functional, "gelu", gelu)
    setattr(torch.nn.functional, "mish", identity)

    setattr(torch.nn.functional, "has_torch_function_unary", F["has_torch_function_unary"])
    setattr(torch.nn.functional, "has_torch_function_variadic", F["has_torch_function_variadic"])
    setattr(torch.nn.functional, "handle_torch_function", F["handle_torch_function"])
    setattr(torch.nn.functional, "torch", F["torch"])

    setattr(torch.Tensor, "__add__", add)
    setattr(torch.Tensor, "__sub__", sub)
    setattr(torch.Tensor, "__mul__", mul)
    setattr(torch.Tensor, "__truediv__", truediv)
    setattr(torch.Tensor, "__matmul__", matmul)
    # setattr(torch.Tensor, "__getitem__", getitem)
    setattr(torch.Tensor, "view", view)
    setattr(torch.Tensor, "permute", permute)
    setattr(torch.Tensor, "transpose", transpose)
    setattr(torch.Tensor, "contiguous", contiguous)
    setattr(torch.Tensor, "to", to)

    yield

    for attr, value in cached_torch_dict.items():
        setattr(torch, attr, value)

    for attr, value in cached_torch_nn_functional_dict.items():
        setattr(torch.nn.functional, attr, value)

    setattr(torch.Tensor, "__add__", torch_add)
    setattr(torch.Tensor, "__sub__", torch_sub)
    setattr(torch.Tensor, "__mul__", torch_mul)
    setattr(torch.Tensor, "__truediv__", torch_truediv)
    setattr(torch.Tensor, "__matmul__", torch_matmul)
    # setattr(torch.Tensor, "__getitem__", torch_getitem)
    setattr(torch.Tensor, "view", torch_view)
    setattr(torch.Tensor, "permute", torch_permute)
    setattr(torch.Tensor, "transpose", torch_transpose)
    setattr(torch.Tensor, "contiguous", torch_contiguous)
    setattr(torch.Tensor, "to", torch_to)