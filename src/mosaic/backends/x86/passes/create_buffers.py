from collections import deque
import math

from loguru import logger
from pyrsistent import pvector
from pyimmer import pmap
from toolz import first

from composit.introspection import class_name
from composit.multidigraph import topological_traversal
from composit.numpy.core import Input, get_operands

from mosaic.backends.x86.types import (
    BufferDescriptor,
    ConstantBufferDescriptor,
    Buffer,
)
from mosaic.passes.inspect import format_bytes
from mosaic.tilelab.tile import create_aligned_array, to_tilized_array


def buffer_descriptor_factory():
    buffer_id = 0

    def create_buffer_descriptor(dtype):
        nonlocal buffer_id
        buffer_descriptor = BufferDescriptor(name=f"intermediate_buffer_descriptor_{buffer_id}_{dtype}")
        buffer_id += 1
        return buffer_descriptor

    return create_buffer_descriptor


create_buffer_descriptor = buffer_descriptor_factory()


def normalize_name(name):
    name = name.replace(".", "_")
    return name


def create_constant_input_buffer_descriptor(name, array):
    return ConstantBufferDescriptor(name=normalize_name(name), array=array)


def is_operation_a_no_operation(operation):
    return class_name(operation) in {"reshape"}


def can_operation_reuse_buffer(operation):
    return class_name(operation) not in {"Tilize", "Untilize", "matmul", "mean", "sum", "transpose"}


def try_returning_buffer_descriptor_to_queue(
    graph, node, node_to_users, buffer_descriptor_to_current_node, dtype_to_buffer_descriptor_stack
):
    for predecessor, _ in graph.in_edges(node):
        if isinstance(graph.nodes[predecessor]["operation"], Input):
            continue

        if predecessor not in node_to_users:
            continue

        node_to_users = node_to_users.set(predecessor, node_to_users[predecessor] - 1)
        if node_to_users[predecessor] == 0 and "buffer_descriptors" in graph.nodes[predecessor]:
            buffer_descriptor = first(graph.nodes[predecessor]["buffer_descriptors"])
            if buffer_descriptor_to_current_node[buffer_descriptor] == predecessor:
                dtype = first(graph.nodes[predecessor]["dtypes"])
                buffer_descriptor_stack = dtype_to_buffer_descriptor_stack.setdefault(dtype, deque(maxlen=None))
                buffer_descriptor_stack.append(buffer_descriptor)

    return node_to_users


def propagate_buffer_down(graph, node, node_to_users, buffer_descriptor_to_current_node):
    if graph.out_degree(node) != 1:
        return graph, node_to_users, buffer_descriptor_to_current_node

    (successor,) = graph.successors(node)

    # Only re-use the buffer of the first operand
    # TODO: figure out why buffers of aother operands don't work
    (first_input_node_to_successor, _), *_ = get_operands(graph, successor)
    if first_input_node_to_successor != node:
        return graph, node_to_users, buffer_descriptor_to_current_node

    dtype = first(graph.nodes[node]["dtypes"])
    successor_dtype = first(graph.nodes[successor]["dtypes"])
    if dtype != successor_dtype:
        return graph, node_to_users, buffer_descriptor_to_current_node

    successor_operation = graph.nodes[successor]["operation"]
    if not can_operation_reuse_buffer(successor_operation):
        return graph, node_to_users, buffer_descriptor_to_current_node

    if "buffer_descriptors" in graph.nodes[successor]:
        return graph, node_to_users, buffer_descriptor_to_current_node

    buffer_descriptor = first(graph.nodes[node]["buffer_descriptors"])

    graph = graph.add_node(successor, buffer_descriptors=tuple([buffer_descriptor]))
    buffer_descriptor_to_current_node = buffer_descriptor_to_current_node.set(buffer_descriptor, successor)

    return propagate_buffer_down(graph, successor, node_to_users, buffer_descriptor_to_current_node)


def is_constant_input(operation):
    # TODO: figure out how to specify that the input is constant
    return False


def populate_buffer_descriptors(graph, reuse_buffers=False):
    dtype_to_buffer_descriptor_stack = {}
    node_to_users = pmap()
    buffer_descriptor_to_current_node = pmap()

    for node in graph:
        node_to_users = node_to_users.set(node, graph.out_degree(node))

    for node in topological_traversal(graph):
        operation = graph.nodes[node]["operation"]

        if is_constant_input(operation):
            graph = graph.add_node(
                node,
                buffer_descriptors=tuple(
                    [create_constant_input_buffer_descriptor(node.name, array=operation.initializer_callback())]
                ),
            )
            continue

        elif is_operation_a_no_operation(operation):
            graph = graph.add_node(
                node,
                buffer_descriptors=tuple([first(graph.nodes[first(graph.predecessors(node))]["buffer_descriptors"])]),
            )

        if "buffer_descriptors" in graph.nodes[node]:
            node_to_users = try_returning_buffer_descriptor_to_queue(
                graph, node, node_to_users, buffer_descriptor_to_current_node, dtype_to_buffer_descriptor_stack
            )
            continue

        dtype = first(graph.nodes[node]["dtypes"])
        if reuse_buffers:
            buffer_descriptor_stack = dtype_to_buffer_descriptor_stack.setdefault(dtype, deque(maxlen=None))
            if not buffer_descriptor_stack:
                buffer_descriptor = create_buffer_descriptor(dtype)
            else:
                buffer_descriptor = buffer_descriptor_stack.pop()
        else:
            buffer_descriptor = create_buffer_descriptor(dtype)

        graph = graph.add_node(node, buffer_descriptors=tuple([buffer_descriptor]))
        buffer_descriptor_to_current_node = buffer_descriptor_to_current_node.set(buffer_descriptor, node)
        if reuse_buffers:
            graph, node_to_users, buffer_descriptor_to_current_node = propagate_buffer_down(
                graph, node, node_to_users, buffer_descriptor_to_current_node
            )
        node_to_users = try_returning_buffer_descriptor_to_queue(
            graph, node, node_to_users, buffer_descriptor_to_current_node, dtype_to_buffer_descriptor_stack
        )

    return graph


def iterate_buffer_descriptors_to_nodes(graph):
    buffer_descriptor_to_nodes = {}
    for node, attributes in graph.nodes(data=True):
        buffer_descriptor = first(attributes["buffer_descriptors"])
        nodes = buffer_descriptor_to_nodes.setdefault(buffer_descriptor, [])
        nodes.append(node)
    buffer_descriptor_to_nodes = pmap(
        {buffer_descriptor: pvector(nodes) for buffer_descriptor, nodes in buffer_descriptor_to_nodes.items()}
    )
    return buffer_descriptor_to_nodes


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
        "Total Memory Used: "
        f"{format_bytes(sum(size * dtype.itemsize for size, dtype in buffer_descriptor_to_size.values()))}"
    )
    return buffer_descriptor_to_size


def allocate_buffers(graph):
    buffer_descriptor_to_size = size_buffers(graph)
    buffer_descriptor_to_buffer = {}
    for buffer_descriptor, (size, dtype) in buffer_descriptor_to_size.items():
        array = create_aligned_array((size,), dtype=dtype)
        array[:] = 0
        buffer_descriptor_to_buffer[buffer_descriptor] = Buffer(array=array)
    buffer_descriptor_to_buffer = pmap(buffer_descriptor_to_buffer)
    return buffer_descriptor_to_buffer


def populate_constant_buffers(graph, buffer_descriptor_to_buffer):
    constant_nodes_with_attributes = (
        (node, attributes) for node, attributes in graph.nodes(data=True) if is_constant_input(attributes["operation"])
    )
    for node, attributes in constant_nodes_with_attributes:
        tile_config = first(attributes["tile_configs"])
        buffer_descriptor = first(attributes["buffer_descriptors"])
        buffer = buffer_descriptor_to_buffer[buffer_descriptor]
        buffer.array[:] = to_tilized_array(buffer_descriptor.array, tile_config)
    return buffer_descriptor_to_buffer


def create_buffers(graph, reuse_buffers):
    graph = populate_buffer_descriptors(graph, reuse_buffers=reuse_buffers)
    buffer_descriptor_to_buffer = allocate_buffers(graph)
    buffer_descriptor_to_buffer = populate_constant_buffers(graph, buffer_descriptor_to_buffer)
    return graph, buffer_descriptor_to_buffer
