from collections import deque
from ctypes import cdll
import enum
import math
import pathlib

import toolz

from loguru import logger
from pyrsistent import pmap, pvector, PClass, field
from toolz import first

from composit.introspection import class_name
from composit.multidigraph import topological_traversal, compose_all
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
    tilize,
)
from mosaic.passes.inspect import format_bytes
from mosaic.tilelab.tile import create_aligned_array, create_array_tile_config, from_tilized_array
from mosaic.tilelab.tile_view import propagate_tile_views


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
        if node_to_users[predecessor] == 0 and "buffer_descriptors" in graph.nodes[predecessor]:
            buffer_descriptor = first(graph.nodes[predecessor]["buffer_descriptors"])
            if buffer_descriptor_to_current_node[buffer_descriptor] == predecessor:
                buffer_descriptor_stack.append(buffer_descriptor)

    return node_to_users


def is_instruction_a_no_operation(instruction):
    return class_name(instruction) in {"reshape"}


def can_instruction_reuse_buffer(instruction):
    return class_name(instruction) not in {"matmul", "mean", "sum", "transpose"}


def propagate_buffer_down(graph, node, node_to_users, buffer_descriptor_to_current_node):
    if graph.out_degree(node) != 1:
        return graph, node_to_users, buffer_descriptor_to_current_node

    (successor,) = graph.successors(node)

    successor_instruction = graph.nodes[successor]["instruction"]
    if not can_instruction_reuse_buffer(successor_instruction):
        return graph, node_to_users, buffer_descriptor_to_current_node

    if "buffer_descriptors" in graph.nodes[successor]:
        return graph, node_to_users, buffer_descriptor_to_current_node

    (input_node_to_successor, _), *_ = get_operands(graph, successor)
    buffer_descriptor = first(graph.nodes[input_node_to_successor]["buffer_descriptors"])

    graph = graph.add_node(successor, buffer_descriptors=tuple([buffer_descriptor]))
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
                node,
                buffer_descriptors=tuple([create_constant_input_buffer_descriptor(node.name, array=instruction.array)]),
            )
            continue
        elif isinstance(instruction, Variable):
            graph = graph.add_node(node, buffer_descriptors=tuple([create_variable_input_buffer_descriptor(node.name)]))
            continue
        elif is_instruction_a_no_operation(instruction):
            graph = graph.add_node(
                node,
                buffer_descriptors=tuple([first(graph.nodes[first(graph.predecessors(node))]["buffer_descriptors"])]),
            )

        if "buffer_descriptors" in graph.nodes[node]:
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

        graph = graph.add_node(node, buffer_descriptors=tuple([buffer_descriptor]))
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
        "Total Memory Used: "
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
    buffer_descriptor = first(graph.nodes[node]["buffer_descriptors"])
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


def propagate_array_tile_config(graph, input_var_to_scheme):
    tile_views = propagate_tile_views(graph, inputs=input_var_to_scheme)

    node_output_to_array_tile_config = {
        node_output: create_array_tile_config(tile_view) for node_output, tile_view in tile_views
    }
    return node_output_to_array_tile_config


def generate_and_compile_kernels(graph, test_output_path, node_output_to_array_tile_config):
    node_to_kernel_name = {}
    for node in graph:
        instruction = graph.nodes[node]["instruction"]
        instruction_class_name = class_name(instruction)

        input_array_tile_configs = [
            node_output_to_array_tile_config[(input_node, output_index)]
            for input_node, output_index in get_operands(graph, node)
        ]
        output_array_tile_config = node_output_to_array_tile_config[(node, 0)]

        if instruction_class_name in {"Variable", "Constant"}:
            kwargs = {}
            user_node = first(graph.successors(node))
            if class_name(graph.nodes[user_node]["instruction"]) == "matmul":
                kwargs = dict(transpose_levels={"tile", "atomic"}, order=(1, 0))

            node_to_kernel_name[node] = tilize.generate_kernel(
                test_output_path, output_array_tile_config, graph.nodes[node]["dtypes"][0], **kwargs
            )
        elif instruction_class_name == "matmul":
            _, (input_b_node, _) = get_operands(graph, node)
            input_b_is_a_variable = class_name(graph.nodes[input_b_node]["instruction"]) == "Variable"
            input_b_levels_to_transpose = set()
            use_avx_manually = False
            if input_b_is_a_variable:
                input_b_levels_to_transpose = {"tile", "atomic"}
                use_avx_manually = True

            node_to_kernel_name[node] = matrix_multiplication.generate_kernel(
                test_output_path,
                *input_array_tile_configs,
                output_array_tile_config,
                input_b_levels_to_transpose=input_b_levels_to_transpose,
                use_avx_manually=use_avx_manually,
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
                test_output_path, *input_array_tile_configs, output_array_tile_config, instruction.axes
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


# def compare(buffer_graph, node, buffer_descriptor_to_buffer, node_output_to_array_tile_config, cache):
#     shape = buffer_graph.nodes[node]["shapes"][0]
#     volume = math.prod(shape)
#     buffer = buffer_descriptor_to_buffer[first(buffer_graph.nodes[node]["buffer_descriptors"])]
#     array_tile_config = node_output_to_array_tile_config[(node, 0)]
#     kernel_array = from_tilized_array(buffer.array[:volume], array_tile_config)
#     cache_array = cache[cnp.nn.variable(name=node.name, shape=())]
#     logger.info(f"Comparing: {node.name}")
#
#     allclose = np.allclose(kernel_array, cache_array, atol=1e-4, rtol=1e-5)
#
#     if not allclose:
#         input_array_tile_configs = [
#             node_output_to_array_tile_config[(input_node, 0)] for input_node, _ in get_operands(buffer_graph, node)
#         ]
#         logger.info(input_array_tile_configs)
#         logger.info(kernel_array.shape)
#         logger.info(cache_array.shape)
#         logger.info(kernel_array)
#         logger.info(cache_array)
#
#     assert allclose


def initialize_variable_buffers(buffer_graph, inputs, node_to_run_kernel, buffer_descriptor_to_buffer):
    for input_var, array in inputs.items():
        input_node = input_var.node
        buffer = buffer_descriptor_to_buffer[first(buffer_graph.nodes[input_node]["buffer_descriptors"])]
        run_kernel = node_to_run_kernel[input_node]
        run_kernel(cast_numpy_array_to_pointer(array.flatten()), buffer.data())


class Model(PClass):
    buffer_graph = field()
    node_to_run_kernel = field()
    buffer_descriptor_to_buffer = field()
    node_output_to_array_tile_config = field()


def evaluate_mosaic_model(model, output_var, inputs):
    initialize_variable_buffers(model.buffer_graph, inputs, model.node_to_run_kernel, model.buffer_descriptor_to_buffer)

    nodes_to_evaluate = filter(
        lambda node: model.buffer_graph.in_degree(node) > 0 and node in model.node_to_run_kernel,
        topological_traversal(model.buffer_graph),
    )

    for node in nodes_to_evaluate:
        input_buffers = [
            model.buffer_descriptor_to_buffer[first(model.buffer_graph.nodes[input_node]["buffer_descriptors"])]
            for input_node, _ in get_operands(model.buffer_graph, node)
        ]
        output_buffer = model.buffer_descriptor_to_buffer[first(model.buffer_graph.nodes[node]["buffer_descriptors"])]

        input_pointers = [input_buffer.data() for input_buffer in input_buffers]
        output_pointer = output_buffer.data()

        run_kernel = model.node_to_run_kernel[node]
        run_kernel(*input_pointers, output_pointer)

    output_node = output_var.node
    array_tile_config = model.node_output_to_array_tile_config[(output_node, 0)]
    buffer = model.buffer_descriptor_to_buffer[first(model.buffer_graph.nodes[output_node]["buffer_descriptors"])]
    return from_tilized_array(buffer.array, array_tile_config)


def compile_to_mosaic_model(
    *output_vars,
    input_var_to_scheme,
    output_path,
    reuse_buffers,
):
    graph = compose_all(*tuple(output_var.graph for output_var in output_vars))
    buffer_graph = populate_buffer_descriptors(graph, reuse_buffers=reuse_buffers)
    buffer_descriptor_to_buffer = allocate_buffers(buffer_graph)

    # from composit.multidigraph import visualize_graph
    # visualize_graph(buffer_graph, visualize_node=visualize_node, timeout=5)

    node_output_to_array_tile_config = propagate_array_tile_config(buffer_graph, input_var_to_scheme)
    node_to_run_kernel = generate_and_compile_kernels(buffer_graph, output_path, node_output_to_array_tile_config)

    mosaic_model = Model(
        buffer_graph=buffer_graph,
        node_to_run_kernel=node_to_run_kernel,
        buffer_descriptor_to_buffer=buffer_descriptor_to_buffer,
        node_output_to_array_tile_config=node_output_to_array_tile_config,
    )

    return mosaic_model
