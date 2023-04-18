import enum
from collections import deque
import math

import numpy as np
import pytest
import toolz

import transformers
from loguru import logger
from pyrsistent import pmap, pvector, PClass, field, pset

import composit as cnp
from composit.multidigraph import compose_all, topological_traversal
from composit.nn import Variable
from composit.numpy.core import Constant
from mosaic.backends.ctypes import cast_numpy_array_to_pointer
from mosaic.passes.inspect import format_bytes

from model_zoo.bert import (
    create_bert_config,
    functional_bert,
    convert_parameters_to_numpy,
)
from mosaic.tilelab.tile import create_aligned_array
from mosaic.tilelab.tile_view import propagate_tile_views
from mosaic.tilelab.tilization_level import TilizationLevel


class BufferType(enum.Enum):
    ConstantInput = enum.auto()
    VariableInput = enum.auto()
    Intermediate = enum.auto()


class Buffer(PClass):
    name: str = field()
    buffer_type: BufferType = field()

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f"Buffer(name={self.name}, buffer_type={self.buffer_type})"


class BufferManager(PClass):
    graph = field()
    buffer_to_nodes = field()

    @property
    def buffers(self):
        return self.buffer_to_nodes.keys()


def intermediate_buffer_factory():
    buffer_id = 0

    def create_intermediate_buffer():
        nonlocal buffer_id
        buffer = Buffer(name=f"intermediate_buffer_{buffer_id}", buffer_type=BufferType.Intermediate)
        buffer_id += 1
        return buffer

    return create_intermediate_buffer


create_intermediate_buffer = intermediate_buffer_factory()


def create_constant_input_buffer(name):
    return Buffer(name=name, buffer_type=BufferType.ConstantInput)


def create_variable_input_buffer(name):
    return Buffer(name=name, buffer_type=BufferType.VariableInput)


def try_returning_buffer_to_queue(graph, node, node_to_users, buffer_to_current_node, buffer_stack):
    for predecessor, _ in graph.in_edges(node):
        if isinstance(graph.nodes[predecessor]["instruction"], (Constant, Variable)):
            continue

        if predecessor not in node_to_users:
            continue

        node_to_users = node_to_users.set(predecessor, node_to_users[predecessor] - 1)
        if node_to_users[predecessor] == 0 and "buffer" in graph.nodes[predecessor]:
            buffer = graph.nodes[predecessor]["buffer"]
            if buffer_to_current_node[buffer] == predecessor:
                buffer_stack.append(buffer)

    return node_to_users


def can_instruction_reuse_buffer(instruction):
    return type(instruction).__name__ not in {"matmul"}


def propagate_buffer_down(graph, node, node_to_users, buffer_to_current_node):
    if graph.out_degree(node) != 1:
        return graph, node_to_users, buffer_to_current_node

    (successor,) = graph.successors(node)

    successor_instruction = graph.nodes[successor]["instruction"]
    if not can_instruction_reuse_buffer(successor_instruction):
        return graph, node_to_users, buffer_to_current_node

    if "buffer" in graph.nodes[successor]:
        return graph, node_to_users, buffer_to_current_node

    buffer = graph.nodes[node]["buffer"]
    graph = graph.add_node(successor, buffer=buffer)
    buffer_to_current_node = buffer_to_current_node.set(buffer, successor)

    return propagate_buffer_down(graph, successor, node_to_users, buffer_to_current_node)


def create_buffer_graph(*outputs):
    buffer_stack = deque(maxlen=None)
    node_to_users = pmap()
    buffer_to_current_node = pmap()

    graph = compose_all(*(output.graph for output in outputs))
    for node in graph:
        node_to_users = node_to_users.set(node, graph.out_degree(node))

    for node in topological_traversal(graph):
        instruction = graph.nodes[node]["instruction"]
        if isinstance(instruction, Constant):
            graph = graph.add_node(node, buffer=create_constant_input_buffer(node.name))
            continue
        elif isinstance(instruction, Variable):
            graph = graph.add_node(node, buffer=create_variable_input_buffer(node.name))
            continue

        if "buffer" in graph.nodes[node]:
            node_to_users = try_returning_buffer_to_queue(
                graph, node, node_to_users, buffer_to_current_node, buffer_stack
            )
            continue

        if not buffer_stack:
            buffer = create_intermediate_buffer()
        else:
            buffer = buffer_stack.pop()

        graph = graph.add_node(node, buffer=buffer)
        buffer_to_current_node = buffer_to_current_node.set(buffer, node)
        graph, node_to_users, buffer_to_current_node = propagate_buffer_down(
            graph, node, node_to_users, buffer_to_current_node
        )
        node_to_users = try_returning_buffer_to_queue(graph, node, node_to_users, buffer_to_current_node, buffer_stack)

    return graph


def size_buffers(graph):
    buffer_to_size = {}
    for buffer, nodes in iterate_buffer_to_nodes(graph).items():
        for node in nodes:
            attributes = graph.nodes[node]
            shapes = attributes["shapes"]
            assert len(shapes) == 1
            shape = shapes[0]

            dtypes = attributes["dtypes"]
            assert len(dtypes) == 1
            dtype = dtypes[0]

            num_bytes = math.prod(shape) * dtype.itemsize
            current_num_bytes = buffer_to_size.setdefault(buffer, 0)

            if num_bytes > current_num_bytes:
                buffer_to_size[buffer] = num_bytes

    buffer_to_size = pmap(buffer_to_size)
    logger.info(f"Total Memory used: {format_bytes(sum(buffer_to_size.values()))}")
    return buffer_to_size


def allocate_buffers(graph):
    buffer_to_size = size_buffers(graph)
    buffer_to_memory = {}
    for buffer, size in buffer_to_size.items():
        logger.info(f"{buffer} : {size}")
        array = create_aligned_array((size,), dtype=np.uint8)
        array[:] = 0
        memory = cast_numpy_array_to_pointer(array)
        buffer_to_memory[buffer] = memory
    buffer_to_memory = pmap(buffer_to_memory)
    return buffer_to_memory


def iterate_buffers(graph):
    buffers = set()
    for node, attributes in graph.nodes(data=True):
        buffers.add(attributes["buffer"])
    buffers = pset(buffers)
    return buffers


def iterate_buffer_to_nodes(graph):
    buffer_to_nodes = {}
    for node, attributes in graph.nodes(data=True):
        nodes = buffer_to_nodes.setdefault(attributes["buffer"], [])
        nodes.append(node)
    buffer_to_nodes = pmap({buffer: pvector(nodes) for buffer, nodes in buffer_to_nodes.items()})
    return buffer_to_nodes


@toolz.memoize
def create_buffer_to_color(graph):
    colors = [
        "red",
        "blue",
        "green",
        "yellow",
        "purple",
        "orange",
    ]

    return {
        buffer: color
        for buffer, color in zip(
            filter(lambda buffer: buffer.buffer_type == BufferType.Intermediate, iterate_buffer_to_nodes(graph)), colors
        )
    }


def visualize_node(graphviz_graph, graph, node):
    buffer_to_color = create_buffer_to_color(graph)
    buffer = graph.nodes[node]["buffer"]
    if buffer.buffer_type == BufferType.Intermediate:
        color = buffer_to_color[buffer]
        fontcolor = {"yellow": "black"}.get(color, "white")
        style = "filled"
    elif buffer.buffer_type == BufferType.ConstantInput:
        color = "black"
        fontcolor = "white"
        style = "filled"
    elif buffer.buffer_type == BufferType.VariableInput:
        color = "black"
        fontcolor = "black"
        style = "solid"
    else:
        raise ValueError("Unrecognized BufferType")
    graphviz_graph.node(node.name, label=f"{node}", color=color, style=style, fontcolor=fontcolor)


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("num_encoders", [3])
@pytest.mark.parametrize("sequence_size", [32])
@pytest.mark.parametrize("num_attention_heads", [4])
@pytest.mark.parametrize("head_size", [48])
@pytest.mark.parametrize("vocab_size", [16])
def test_bert(
    batch_size,
    num_encoders,
    sequence_size,
    num_attention_heads,
    head_size,
    vocab_size,
):
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
        cnp.nn.variable(name=name, shape=value.shape, dtype=np.float16): value
        for name, value in convert_parameters_to_numpy(transformers_model).items()
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

    graph = create_buffer_graph(model)
    buffer_to_memory = allocate_buffers(graph)

    for node in graph:
        input_pointers = [buffer_to_memory[graph.nodes[input_node]["buffer"]] for input_node, _ in graph.in_edges(node)]
        output_pointer = buffer_to_memory[graph.nodes[node]["buffer"]]
        logger.info(f"node={node}, num_inputs={len(input_pointers)}")
