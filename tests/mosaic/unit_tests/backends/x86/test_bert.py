from collections import deque
import enum
import math
import pathlib
from ctypes import cdll

import numpy as np
import pytest
import toolz
import torch

import transformers
from loguru import logger
from pyrsistent import pmap, pvector, PClass, field, pset
from toolz import first

import composit as cnp
from composit.hash import deterministic_hash
from composit.introspection import class_name
from composit.multidigraph import topological_traversal
from composit.nn import Variable
from composit.numpy.core import Constant, get_operands
from mosaic.backends.ctypes import cast_numpy_array_to_pointer
from mosaic.backends.x86.compile import compile_shared_library
from mosaic.backends.x86.kernels import (
    matrix_multiplication,
    unary_operation,
    binary_operation,
    reduce,
    transpose,
    embedding,
)
from mosaic.passes.inspect import format_bytes
from mosaic.tilelab.tile import create_aligned_array, create_array_tile_config, to_flat_array, from_flat_array
from mosaic.tilelab.tile_view import propagate_tile_views, TileLevel


from model_zoo.bert import (
    create_bert_config,
    functional_bert,
    convert_parameters_to_numpy,
    create_random_long,
)

FILE_DIR = pathlib.Path(__file__).parent.resolve()


class BufferType(enum.Enum):
    ConstantInput = enum.auto()
    VariableInput = enum.auto()
    Intermediate = enum.auto()


class BufferDescriptor(PClass):
    name: str = field()
    buffer_type: BufferType = field()

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f"BufferDescriptor(name={self.name}, buffer_type={self.buffer_type})"


class ConstantBufferDescriptor(PClass):
    name: str = field()
    buffer_type: BufferType = field()
    array = field()

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f"ConstantBufferDescriptor(name={self.name}, buffer_type={self.buffer_type}, array={self.array})"


class BufferManager(PClass):
    graph = field()
    buffer_descriptors_to_nodes = field()


def intermediate_buffer_descriptor_factory():
    buffer_id = 0

    def create_intermediate_buffer_descriptor():
        nonlocal buffer_id
        buffer_descriptor = BufferDescriptor(
            name=f"intermediate_buffer_descriptor_{buffer_id}", buffer_type=BufferType.Intermediate
        )
        buffer_id += 1
        return buffer_descriptor

    return create_intermediate_buffer_descriptor


create_intermediate_buffer_descriptor = intermediate_buffer_descriptor_factory()


def create_constant_input_buffer_descriptor(name, array):
    return ConstantBufferDescriptor(name=name, buffer_type=BufferType.ConstantInput, array=array)


def create_variable_input_buffer_descriptor(name):
    return BufferDescriptor(name=name, buffer_type=BufferType.VariableInput)


def try_returning_buffer_descriptor_to_queue(
    graph, node, node_to_users, buffer_descriptor_to_current_node, buffer_descriptor_stack
):
    for predecessor, _ in graph.in_edges(node):
        if isinstance(graph.nodes[predecessor]["instruction"], (Constant, Variable)):
            continue

        if predecessor not in node_to_users:
            continue

        node_to_users = node_to_users.set(predecessor, node_to_users[predecessor] - 1)
        if node_to_users[predecessor] == 0 and "buffer_descriptor" in graph.nodes[predecessor]:
            buffer_descriptor = graph.nodes[predecessor]["buffer_descriptor"]
            if buffer_descriptor_to_current_node[buffer_descriptor] == predecessor:
                buffer_descriptor_stack.append(buffer_descriptor)

    return node_to_users


def is_instruction_a_no_operation(instruction):
    return class_name(instruction) in {"reshape"}


def can_instruction_reuse_buffer(instruction):
    return class_name(instruction) not in {"matmul"}


def propagate_buffer_down(graph, node, node_to_users, buffer_descriptor_to_current_node):
    if graph.out_degree(node) != 1:
        return graph, node_to_users, buffer_descriptor_to_current_node

    (successor,) = graph.successors(node)

    successor_instruction = graph.nodes[successor]["instruction"]
    if not can_instruction_reuse_buffer(successor_instruction):
        return graph, node_to_users, buffer_descriptor_to_current_node

    if "buffer_descriptor" in graph.nodes[successor]:
        return graph, node_to_users, buffer_descriptor_to_current_node

    buffer_descriptor = graph.nodes[node]["buffer_descriptor"]
    graph = graph.add_node(successor, buffer_descriptor=buffer_descriptor)
    buffer_descriptor_to_current_node = buffer_descriptor_to_current_node.set(buffer_descriptor, successor)

    return propagate_buffer_down(graph, successor, node_to_users, buffer_descriptor_to_current_node)


def populate_buffer_descriptors(graph, reuse_buffers=False):
    buffer_descriptor_stack = deque(maxlen=None)
    node_to_users = pmap()
    buffer_descriptor_to_current_node = pmap()

    for node in graph:
        node_to_users = node_to_users.set(node, graph.out_degree(node))

    for node in topological_traversal(graph):
        instruction = graph.nodes[node]["instruction"]
        if isinstance(instruction, Constant):
            graph = graph.add_node(
                node, buffer_descriptor=create_constant_input_buffer_descriptor(node.name, array=instruction.array)
            )
            continue
        elif isinstance(instruction, Variable):
            graph = graph.add_node(node, buffer_descriptor=create_variable_input_buffer_descriptor(node.name))
            continue
        elif is_instruction_a_no_operation(instruction):
            graph = graph.add_node(
                node, buffer_descriptor=graph.nodes[first(graph.predecessors(node))]["buffer_descriptor"]
            )

        if "buffer_descriptor" in graph.nodes[node]:
            node_to_users = try_returning_buffer_descriptor_to_queue(
                graph, node, node_to_users, buffer_descriptor_to_current_node, buffer_descriptor_stack
            )
            continue

        if reuse_buffers:
            if not buffer_descriptor_stack:
                buffer_descriptor = create_intermediate_buffer_descriptor()
            else:
                buffer_descriptor = buffer_descriptor_stack.pop()
        else:
            buffer_descriptor = create_intermediate_buffer_descriptor()

        graph = graph.add_node(node, buffer_descriptor=buffer_descriptor)
        buffer_descriptor_to_current_node = buffer_descriptor_to_current_node.set(buffer_descriptor, node)
        if reuse_buffers:
            graph, node_to_users, buffer_descriptor_to_current_node = propagate_buffer_down(
                graph, node, node_to_users, buffer_descriptor_to_current_node
            )
        node_to_users = try_returning_buffer_descriptor_to_queue(
            graph, node, node_to_users, buffer_descriptor_to_current_node, buffer_descriptor_stack
        )

    return graph


def size_buffers(graph):
    buffer_descriptor_to_size = {}
    for buffer_descriptor, nodes in iterate_buffer_descriptors_to_nodes(graph).items():
        for node in nodes:
            attributes = graph.nodes[node]
            shapes = attributes["shapes"]
            assert len(shapes) == 1
            shape = shapes[0]

            dtypes = attributes["dtypes"]
            assert len(dtypes) == 1
            dtype = dtypes[0]

            num_bytes = math.prod(shape)
            current_num_bytes, current_dtype = buffer_descriptor_to_size.setdefault(buffer_descriptor, (0, None))
            if current_dtype is not None:
                assert current_dtype == dtype

            if num_bytes > current_num_bytes:
                buffer_descriptor_to_size[buffer_descriptor] = (num_bytes, dtype)

    buffer_descriptor_to_size = pmap(buffer_descriptor_to_size)
    logger.info(
        "Total Memory used: "
        f"{format_bytes(sum(size * dtype.itemsize for size, dtype in buffer_descriptor_to_size.values()))}"
    )
    return buffer_descriptor_to_size


class Buffer(PClass):
    array = field()

    def data(self):
        return cast_numpy_array_to_pointer(self.array)


def allocate_buffers(graph):
    buffer_descriptor_to_size = size_buffers(graph)
    buffer_descriptor_to_buffer = {}
    for buffer_descriptor, (size, dtype) in buffer_descriptor_to_size.items():
        array = create_aligned_array((size,), dtype=dtype)
        if buffer_descriptor.buffer_type == BufferType.ConstantInput:
            array[:] = buffer_descriptor.array
        else:
            array[:] = 0
        buffer_descriptor_to_buffer[buffer_descriptor] = Buffer(array=array)
    buffer_descriptor_to_buffer = pmap(buffer_descriptor_to_buffer)
    return buffer_descriptor_to_buffer


def iterate_buffer_descriptors(graph):
    buffer_descriptors = set()
    for node, attributes in graph.nodes(data=True):
        buffer_descriptors.add(attributes["buffer_descriptor"])
    buffer_descriptors = pset(buffer_descriptors)
    return buffer_descriptors


def iterate_buffer_descriptors_to_nodes(graph):
    buffer_descriptor_to_nodes = {}
    for node, attributes in graph.nodes(data=True):
        nodes = buffer_descriptor_to_nodes.setdefault(attributes["buffer_descriptor"], [])
        nodes.append(node)
    buffer_descriptor_to_nodes = pmap(
        {buffer_descriptor: pvector(nodes) for buffer_descriptor, nodes in buffer_descriptor_to_nodes.items()}
    )
    return buffer_descriptor_to_nodes


@toolz.memoize
def create_buffer_descriptor_to_color_and_style(graph):
    nodes = filter(
        lambda buffer_descriptor: buffer_descriptor.buffer_type == BufferType.Intermediate,
        iterate_buffer_descriptors_to_nodes(graph),
    )
    nodes = list(nodes)

    import colorsys

    hsv_tuples = [(i * 1.0 / 100, 0.5, 0.5) for i in range(len(nodes))]
    colors = ["#%02x%02x%02x" % tuple(map(lambda hsv: int(hsv * 255), colorsys.hsv_to_rgb(*hsv))) for hsv in hsv_tuples]

    return {buffer_descriptor: (color, "filled") for i, (buffer_descriptor, color) in enumerate(zip(nodes, colors))}


def visualize_node(graphviz_graph, graph, node):
    buffer_descriptor_to_color_and_style = create_buffer_descriptor_to_color_and_style(graph)
    buffer_descriptor = graph.nodes[node]["buffer_descriptor"]
    if buffer_descriptor.buffer_type == BufferType.Intermediate:
        color, style = buffer_descriptor_to_color_and_style[buffer_descriptor]
        fontcolor = "white"
    elif buffer_descriptor.buffer_type == BufferType.ConstantInput:
        color = "black"
        fontcolor = "white"
        style = "filled"
    elif buffer_descriptor.buffer_type == BufferType.VariableInput:
        color = "black"
        fontcolor = "black"
        style = "solid"
    else:
        raise ValueError("Unrecognized BufferType")
    graphviz_graph.node(
        node.name, label=f"{node} @ {buffer_descriptor.name}", color=color, style=style, fontcolor=fontcolor
    )


def create_tile_shape(var):
    shape = var.shape
    if len(shape) == 3:
        return (1, 32, 32)
    elif len(shape) == 2:
        if "embeddings.weight" in var.name:
            return (1, 32)
        return min(shape[0], 32), min(shape[1], 32)
    elif len(shape) == 1:
        return (min(shape[0], 32),)
    else:
        return ()


def propagate_array_tile_config(graph, input_var_to_scheme):
    tile_views = propagate_tile_views(graph, inputs=input_var_to_scheme)

    node_output_to_array_tile_config = {
        node_output: create_array_tile_config(tile_view) for node_output, tile_view in tile_views
    }
    return node_output_to_array_tile_config


def generate_and_compile_kernels(graph, test_output_path, node_output_to_array_tile_config):
    nodes = filter(lambda node: graph.in_degree(node) > 0, graph)

    node_to_kernel_name = {}
    for node in nodes:
        instruction = graph.nodes[node]["instruction"]
        instruction_class_name = class_name(instruction)

        input_array_tile_configs = [
            node_output_to_array_tile_config[(input_node, 0)] for input_node, _ in get_operands(graph, node)
        ]
        output_array_tile_config = node_output_to_array_tile_config[(node, 0)]

        if instruction_class_name == "matmul":
            node_to_kernel_name[node] = matrix_multiplication.generate_kernel(
                test_output_path, *input_array_tile_configs, input_b_levels_to_transpose=None, use_avx_manually=True
            )
        elif instruction_class_name in {"exp", "sqrt", "gelu"}:
            node_to_kernel_name[node] = unary_operation.generate_kernel(
                test_output_path, *input_array_tile_configs, instruction_class_name
            )
        elif instruction_class_name in {"add", "subtract", "divide", "multiply"}:
            node_to_kernel_name[node] = binary_operation.generate_kernel(
                test_output_path, *input_array_tile_configs, instruction_class_name
            )
        elif instruction_class_name in {"reshape"}:
            node_to_kernel_name[node] = None
        elif instruction_class_name in {"sum", "mean", "max"}:
            node_to_kernel_name[node] = reduce.generate_kernel(
                test_output_path,
                *input_array_tile_configs,
                output_array_tile_config,
                instruction_class_name,
            )
        elif instruction_class_name in {"embedding"}:
            node_to_kernel_name[node] = node_to_kernel_name[node] = node_to_kernel_name[
                node
            ] = embedding.generate_kernel(test_output_path, output_array_tile_config)
        elif instruction_class_name in {"transpose"}:
            node_to_kernel_name[node] = node_to_kernel_name[node] = transpose.generate_kernel(
                test_output_path,
                *input_array_tile_configs,
                output_array_tile_config,
                instruction.axes,
            )
        else:
            raise NotImplementedError(f"There is no kernel implementation for {instruction_class_name}")

    kernel_name_to_run_kernel = {}
    for kernel_name in set(node_to_kernel_name.values()):
        if kernel_name is None:
            kernel_name_to_run_kernel[kernel_name] = lambda *_: None
        else:
            shared_library_file = compile_shared_library(test_output_path, kernel_name)
            shared_library = cdll.LoadLibrary(shared_library_file)
            run_kernel = getattr(shared_library, kernel_name)
            kernel_name_to_run_kernel[kernel_name] = run_kernel

    node_to_run_kernel = {
        node: kernel_name_to_run_kernel[kernel_name] for node, kernel_name in node_to_kernel_name.items()
    }
    return pmap(node_to_run_kernel)


# def compare(graph, node, buffer_descriptor_to_buffer, node_output_to_array_tile_config, cache):
#     shape = graph.nodes[node]["shapes"][0]
#     volume = math.prod(shape)
#     buffer = buffer_descriptor_to_buffer[graph.nodes[node]["buffer_descriptor"]]
#     array_tile_config = node_output_to_array_tile_config[(node, 0)]
#     kernel_array = from_flat_array(buffer.array[:volume], array_tile_config)
#     cache_array = cache[cnp.nn.variable(name=node.name, shape=())]
#     logger.info(f"Comparing: {node.name}")
#
#     allclose = np.allclose(
#         kernel_array,
#         cache_array,
#         atol=1e-4,
#         rtol=1e-5,
#     )
#
#     if not allclose:
#         input_array_tile_configs = [
#             node_output_to_array_tile_config[(input_node, 0)] for input_node, _ in get_operands(graph, node)
#         ]
#         logger.info(input_array_tile_configs)
#         logger.info(kernel_array.shape)
#         logger.info(cache_array.shape)
#         logger.info(kernel_array)
#         logger.info(cache_array)
#
#     assert allclose


def initialize_variable_buffers(graph, inputs, buffer_descriptor_to_buffer, node_output_to_array_tile_config):
    for input_var, array in inputs.items():
        input_node = input_var.node
        array_tile_config = node_output_to_array_tile_config[(input_node, 0)]
        buffer = buffer_descriptor_to_buffer[graph.nodes[input_node]["buffer_descriptor"]]
        buffer.array[:] = to_flat_array(array, array_tile_config)


def evaluate(
    output_var, graph, inputs, node_to_run_kernel, buffer_descriptor_to_buffer, node_output_to_array_tile_config
):
    initialize_variable_buffers(graph, inputs, buffer_descriptor_to_buffer, node_output_to_array_tile_config)

    nodes_to_evaluate = filter(
        lambda node: graph.in_degree(node) > 0 and node in node_to_run_kernel, topological_traversal(graph)
    )

    for node in nodes_to_evaluate:
        input_buffers = [
            buffer_descriptor_to_buffer[graph.nodes[input_node]["buffer_descriptor"]]
            for input_node, _ in get_operands(graph, node)
        ]
        output_buffer = buffer_descriptor_to_buffer[graph.nodes[node]["buffer_descriptor"]]

        input_pointers = [input_buffer.data() for input_buffer in input_buffers]
        output_pointer = output_buffer.data()

        run_kernel = node_to_run_kernel[node]
        run_kernel(*input_pointers, output_pointer)

    output_node = output_var.node
    array_tile_config = node_output_to_array_tile_config[(output_node, 0)]
    buffer = buffer_descriptor_to_buffer[graph.nodes[output_node]["buffer_descriptor"]]
    return from_flat_array(buffer.array, array_tile_config)


@pytest.mark.parametrize("num_inputs", [1])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("num_encoders", [3])
@pytest.mark.parametrize("sequence_size", [32])
@pytest.mark.parametrize("num_attention_heads", [4])
@pytest.mark.parametrize("head_size", [32])
@pytest.mark.parametrize("vocab_size", [16])
@pytest.mark.parametrize("reuse_buffers", [False])
def test_bert(
    request,
    num_inputs,
    batch_size,
    num_encoders,
    sequence_size,
    num_attention_heads,
    head_size,
    vocab_size,
    reuse_buffers,
):
    np.random.seed(0)
    torch.manual_seed(0)

    test_name = request.node.name
    test_output_path = FILE_DIR / "test_output" / str(deterministic_hash(test_name))
    test_output_path.mkdir(parents=True, exist_ok=True)

    config = create_bert_config(
        num_encoders=num_encoders,
        num_attention_heads=num_attention_heads,
        head_size=head_size,
        vocab_size=vocab_size,
    )

    transformers_model = transformers.models.bert.modeling_bert.BertModel(config)

    input_ids_var = cnp.nn.variable(name="input_ids", shape=(batch_size, sequence_size), dtype=np.int64)
    token_type_ids_var = cnp.nn.variable(name="token_type_ids", shape=(batch_size, sequence_size), dtype=np.int64)
    parameters = {
        cnp.nn.variable(name=name, shape=value.shape, dtype=np.float32): value
        for name, value in convert_parameters_to_numpy(transformers_model).items()
        if "position_embeddings" not in name and "pooler" not in name
    }

    with cnp.nn.module.disable_modules():
        model = functional_bert(
            input_ids_var,
            token_type_ids_var,
            None,
            {var.node.name: var for var in parameters.keys()},
            num_encoders=num_encoders,
            sequence_size=sequence_size,
            num_attention_heads=num_attention_heads,
            head_size=head_size,
        )

    input_var_to_scheme = {
        var: [
            TileLevel(level_name="tile", tile_shape=create_tile_shape(var)),
        ]
        for var in [input_ids_var, token_type_ids_var] + list(parameters.keys())
    }

    graph = model.graph

    graph = populate_buffer_descriptors(graph, reuse_buffers=reuse_buffers)
    buffer_descriptor_to_buffer = allocate_buffers(graph)
    # visualize_graph(graph, visualize_node=visualize_node)

    node_output_to_array_tile_config = propagate_array_tile_config(graph, input_var_to_scheme)
    node_to_run_kernel = generate_and_compile_kernels(graph, test_output_path, node_output_to_array_tile_config)

    for _ in range(num_inputs):
        input_ids = create_random_long((batch_size, sequence_size), minimum=0, maximum=vocab_size)
        token_type_ids = np.zeros((batch_size, sequence_size), dtype=np.int64)

        inputs = {
            input_ids_var: input_ids,
            token_type_ids_var: token_type_ids,
            **parameters,
        }

        golden_output, cache = cnp.nn.evaluate(
            model,
            inputs=inputs,
            return_cache=True,
        )

        output = evaluate(
            model,
            graph,
            inputs=inputs,
            node_to_run_kernel=node_to_run_kernel,
            buffer_descriptor_to_buffer=buffer_descriptor_to_buffer,
            node_output_to_array_tile_config=node_output_to_array_tile_config,
        )

        assert np.allclose(output, golden_output, atol=1e-4, rtol=1e-5)
