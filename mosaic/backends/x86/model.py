from collections import deque
from ctypes import cdll
import enum
import math
import pathlib

import numpy as np
import toolz

from loguru import logger
from pyrsistent import pmap, pvector, PClass, field
from toolz import first

import codegen as c
from composit.introspection import class_name
from composit.multidigraph import topological_traversal, compose_all, MultiDiGraph
from composit.nn import Variable
from composit.numpy.core import Constant, get_operands
from composit.persistent_array import Node
from mosaic.backends.ctypes import cast_numpy_array_to_pointer
from mosaic.backends.x86.compile import compile_shared_library
from mosaic.backends.x86.constants import MEMORY_ALIGNMENT
from mosaic.backends.x86.kernels import (
    matrix_multiplication,
    unary_operation,
    binary_operation,
    reduce,
    transpose,
    embedding,
    tilize,
    untilize,
)
from mosaic.passes.inspect import format_bytes
from mosaic.tilelab.tile import create_aligned_array, create_array_tile_config, to_tilized_array
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

    def __eq__(self, other: "ConstantBufferDescriptor"):
        return (
            self.name == other.name and self.buffer_type == other.buffer_type and np.allclose(self.array, other.array)
        )

    def __repr__(self):
        return f"ConstantBufferDescriptor(name={self.name}, buffer_type={self.buffer_type}, array={self.array})"


class BufferManager(PClass):
    graph = field()
    buffer_descriptors_to_nodes = field()


def intermediate_buffer_descriptor_factory():
    buffer_id = 0

    def create_intermediate_buffer_descriptor(dtype):
        nonlocal buffer_id
        buffer_descriptor = BufferDescriptor(
            name=f"intermediate_buffer_descriptor_{buffer_id}_{dtype}", buffer_type=BufferType.Intermediate
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
    graph, node, node_to_users, buffer_descriptor_to_current_node, dtype_to_buffer_descriptor_stack
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
                dtype = first(graph.nodes[predecessor]["dtypes"])
                buffer_descriptor_stack = dtype_to_buffer_descriptor_stack.setdefault(dtype, deque(maxlen=None))
                buffer_descriptor_stack.append(buffer_descriptor)

    return node_to_users


def is_instruction_a_no_operation(instruction):
    return class_name(instruction) in {"reshape"}


def can_instruction_reuse_buffer(instruction):
    return class_name(instruction) not in {"Tilize", "Untilize", "matmul", "mean", "sum", "transpose"}


def propagate_buffer_down(graph, node, node_to_users, buffer_descriptor_to_current_node):
    if graph.out_degree(node) != 1:
        return graph, node_to_users, buffer_descriptor_to_current_node

    (successor,) = graph.successors(node)

    dtype = first(graph.nodes[node]["dtypes"])
    successor_dtype = first(graph.nodes[successor]["dtypes"])
    if dtype != successor_dtype:
        return graph, node_to_users, buffer_descriptor_to_current_node

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
    dtype_to_buffer_descriptor_stack = {}
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
                graph, node, node_to_users, buffer_descriptor_to_current_node, dtype_to_buffer_descriptor_stack
            )
            continue

        dtype = first(graph.nodes[node]["dtypes"])
        if reuse_buffers:
            buffer_descriptor_stack = dtype_to_buffer_descriptor_stack.setdefault(dtype, deque(maxlen=None))
            if not buffer_descriptor_stack:
                buffer_descriptor = create_intermediate_buffer_descriptor(dtype)
            else:
                buffer_descriptor = buffer_descriptor_stack.pop()
        else:
            buffer_descriptor = create_intermediate_buffer_descriptor(dtype)

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
        array[:] = 0
        buffer_descriptor_to_buffer[buffer_descriptor] = Buffer(array=array)
    buffer_descriptor_to_buffer = pmap(buffer_descriptor_to_buffer)
    return buffer_descriptor_to_buffer


def populate_constant_buffers(graph, buffer_descriptor_to_buffer, node_output_to_array_tile_config):
    constant_nodes_with_attributes = (
        (node, attributes)
        for node, attributes in graph.nodes(data=True)
        if class_name(attributes["instruction"]) == "Constant"
    )
    for node, attributes in constant_nodes_with_attributes:
        output_index = 0
        array_tile_config = node_output_to_array_tile_config[(node, output_index)]
        buffer_descriptor = first(attributes["buffer_descriptors"])
        buffer = buffer_descriptor_to_buffer[buffer_descriptor]
        buffer.array[:] = to_tilized_array(buffer_descriptor.array, array_tile_config)
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


class Tilize(PClass):
    def __call__(self, input_tensor):
        return input_tensor


class Untilize(PClass):
    def __call__(self, input_tensor):
        return input_tensor


def insert_tilize_and_untilize_instructions(graph, node_output_to_array_tile_config):
    new_graph = MultiDiGraph()
    operand_to_new_operand = {}
    for node in topological_traversal(graph):
        attributes = graph.nodes[node]

        new_graph = new_graph.add_node(node, **attributes)
        operand_to_new_operand[(node, 0)] = (node, 0)

        if class_name(graph.nodes[node]["instruction"]) == "Variable":
            tilize_node = Node(name=f"tilize_{node.name}")
            new_graph = new_graph.add_node(
                tilize_node, instruction=Tilize(), shapes=attributes["shapes"], dtypes=attributes["dtypes"]
            )
            new_graph = new_graph.add_edge(node, tilize_node, source_output_index=0, sink_input_index=0)
            operand_to_new_operand[(node, 0)] = (tilize_node, 0)
            node_output_to_array_tile_config[(tilize_node, 0)] = node_output_to_array_tile_config[(node, 0)]
        elif graph.out_degree(node) == 0:
            untilize_node = Node(name=f"untilize_{node.name}")
            new_graph = new_graph.add_node(
                untilize_node, instruction=Untilize(), shapes=attributes["shapes"], dtypes=attributes["dtypes"]
            )
            new_graph = new_graph.add_edge(node, untilize_node, source_output_index=0, sink_input_index=0)
            node_output_to_array_tile_config[(untilize_node, 0)] = node_output_to_array_tile_config[(node, 0)]

        for source, sink, edge_attributes in graph.in_edges(node, data=True):
            operand = (source, edge_attributes["source_output_index"])
            (new_source, new_source_output_index) = operand_to_new_operand[operand]
            new_graph = new_graph.add_edge(
                new_source,
                node,
                source_output_index=new_source_output_index,
                sink_input_index=edge_attributes["sink_input_index"],
            )

    return new_graph, node_output_to_array_tile_config


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

        if instruction_class_name in {"Constant", "Variable"}:
            continue
        elif instruction_class_name == "Tilize":
            node_to_kernel_name[node] = tilize.generate_kernel_source_file(
                test_output_path,
                output_array_tile_config,
                graph.nodes[node]["dtypes"][0],
            )
        elif instruction_class_name == "Untilize":
            node_to_kernel_name[node] = untilize.generate_kernel_source_file(
                test_output_path,
                output_array_tile_config,
                graph.nodes[node]["dtypes"][0],
            )
        elif instruction_class_name == "matmul":
            node_to_kernel_name[node] = matrix_multiplication.generate_kernel_source_file(
                test_output_path,
                *input_array_tile_configs,
                output_array_tile_config,
                use_avx_manually=True,
            )
        elif instruction_class_name in {"exp", "sqrt", "gelu"}:
            node_to_kernel_name[node] = unary_operation.generate_kernel_source_file(
                test_output_path, *input_array_tile_configs, instruction_class_name
            )
        elif instruction_class_name in {"add", "subtract", "divide", "multiply"}:
            node_to_kernel_name[node] = binary_operation.generate_kernel_source_file(
                test_output_path, *input_array_tile_configs, instruction_class_name
            )
        elif instruction_class_name in {"reshape"}:
            node_to_kernel_name[node] = None
        elif instruction_class_name in {"sum", "mean", "max"}:
            node_to_kernel_name[node] = reduce.generate_kernel_source_file(
                test_output_path,
                *input_array_tile_configs,
                output_array_tile_config,
                instruction_class_name,
            )
        elif instruction_class_name in {"embedding"}:
            node_to_kernel_name[node] = node_to_kernel_name[node] = node_to_kernel_name[
                node
            ] = embedding.generate_kernel_source_file(test_output_path, output_array_tile_config)
        elif instruction_class_name in {"transpose"}:
            node_to_kernel_name[node] = node_to_kernel_name[node] = transpose.generate_kernel_source_file(
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


class ModelWithoutKernelFusion(PClass):
    buffer_graph = field()
    node_to_run_kernel = field()
    buffer_descriptor_to_buffer = field()


def generate_and_compile_run_model(
    graph, test_output_path, node_output_to_array_tile_config, buffer_descriptor_to_buffer
):
    module = c.Module(includes=[], functions=[])
    node_to_kernel_name = {}

    for node, attributes in graph.nodes(data=True):
        instruction = attributes["instruction"]
        instruction_class_name = class_name(instruction)

        input_array_tile_configs = [
            node_output_to_array_tile_config[(input_node, output_index)]
            for input_node, output_index in get_operands(graph, node)
        ]
        output_array_tile_config = node_output_to_array_tile_config[(node, 0)]

        input_dtypes = [
            graph.nodes[input_node]["dtypes"][output_index] for input_node, output_index in get_operands(graph, node)
        ]
        output_dtype = attributes["dtypes"][0]

        if instruction_class_name in {"Constant", "Variable"}:
            continue
        elif instruction_class_name == "Tilize":
            kernel_name, kernel_module = tilize.generate_module(
                input_array_tile_configs,
                output_array_tile_config,
                input_dtypes,
                output_dtype,
            )
        elif instruction_class_name == "Untilize":
            kernel_name, kernel_module = untilize.generate_module(
                input_array_tile_configs,
                output_array_tile_config,
                input_dtypes,
                output_dtype,
            )
        elif instruction_class_name == "matmul":
            kernel_name, kernel_module = matrix_multiplication.generate_module(
                input_array_tile_configs,
                output_array_tile_config,
                input_dtypes,
                output_dtype,
                use_avx_manually=True,
            )
        elif instruction_class_name in {"add", "subtract", "divide", "multiply"}:
            kernel_name, kernel_module = binary_operation.generate_module(
                input_array_tile_configs,
                output_array_tile_config,
                input_dtypes,
                output_dtype,
                instruction_class_name,
            )
        else:
            raise NotImplementedError(f"There is no kernel implementation for {instruction_class_name}")
        module += kernel_module
        node_to_kernel_name[node] = kernel_name

    arguments = []
    buffer_descriptor_to_variable = {}
    for buffer_descriptor in buffer_descriptor_to_buffer:
        FloatPointer = c.Type("float").pointer().restrict().aligned(MEMORY_ALIGNMENT)
        if buffer_descriptor.buffer_type == BufferType.VariableInput:
            FloatPointer = FloatPointer.const()

        variable = c.variable(FloatPointer, buffer_descriptor.name)
        arguments.append(variable)
        buffer_descriptor_to_variable[buffer_descriptor] = variable
    logger.info(arguments)

    statements = []
    for node in filter(
        lambda node: graph.in_degree(node) > 0,
        topological_traversal(graph),
    ):
        input_vars = [
            buffer_descriptor_to_variable[first(graph.nodes[input_node]["buffer_descriptors"])]
            for input_node, _ in get_operands(graph, node)
        ]
        output_var = buffer_descriptor_to_variable[first(graph.nodes[node]["buffer_descriptors"])]

        kernel_name = node_to_kernel_name[node]
        invocation = c.invoke(kernel_name, *input_vars, output_var)
        statements.append(c.Statement(invocation))

    model_name = "run_model"
    module += c.Module(
        includes=[],
        functions=[
            c.Function(
                return_type=c.Type("void"),
                name=c.Identifier(model_name),
                arguments=arguments,
                body=c.block(*statements),
            ).extern_c()
        ],
    )

    module.save((test_output_path / model_name).with_suffix(".cpp"))

    shared_library_file = compile_shared_library(test_output_path, model_name)
    shared_library = cdll.LoadLibrary(shared_library_file)
    run_model = getattr(shared_library, model_name)
    return run_model


class ModelWithKernelFusion(PClass):
    buffer_graph = field()
    run_model = field()
    buffer_descriptor_to_buffer = field()


def compile_to_mosaic_model(
    *output_vars,
    input_var_to_scheme,
    output_path,
    reuse_buffers: bool,
    fuse_kernels: bool = False,
):
    graph = compose_all(*tuple(output_var.graph for output_var in output_vars))
    node_output_to_array_tile_config = propagate_array_tile_config(graph, input_var_to_scheme)
    graph, node_output_to_array_tile_config = insert_tilize_and_untilize_instructions(
        graph, node_output_to_array_tile_config
    )
    buffer_graph = populate_buffer_descriptors(graph, reuse_buffers=reuse_buffers)
    buffer_descriptor_to_buffer = allocate_buffers(buffer_graph)
    buffer_descriptor_to_buffer = populate_constant_buffers(
        buffer_graph, buffer_descriptor_to_buffer, node_output_to_array_tile_config
    )

    # from composit.multidigraph import visualize_graph
    # visualize_graph(buffer_graph, visualize_node=visualize_node, timeout=5)

    if fuse_kernels:
        run_model = generate_and_compile_run_model(
            buffer_graph, output_path, node_output_to_array_tile_config, buffer_descriptor_to_buffer
        )
        return ModelWithKernelFusion(
            buffer_graph=buffer_graph,
            run_model=run_model,
            buffer_descriptor_to_buffer=buffer_descriptor_to_buffer,
        )

    node_to_run_kernel = generate_and_compile_kernels(buffer_graph, output_path, node_output_to_array_tile_config)
    return ModelWithoutKernelFusion(
        buffer_graph=buffer_graph,
        node_to_run_kernel=node_to_run_kernel,
        buffer_descriptor_to_buffer=buffer_descriptor_to_buffer,
    )


def initialize_variable_buffers(buffer_graph, inputs, buffer_descriptor_to_buffer):
    for input_var, array in inputs.items():
        input_node = input_var.node
        buffer = buffer_descriptor_to_buffer[first(buffer_graph.nodes[input_node]["buffer_descriptors"])]
        buffer.array[:] = array.flatten()


def evaluate_mosaic_model_without_kernel_fusion(model: ModelWithoutKernelFusion, output_var, inputs):
    initialize_variable_buffers(model.buffer_graph, inputs, model.buffer_descriptor_to_buffer)

    nodes_to_evaluate = filter(
        lambda node: model.buffer_graph.in_degree(node) > 0, topological_traversal(model.buffer_graph)
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

    output_node = first(model.buffer_graph.successors(output_var.node))
    buffer_descriptor = first(model.buffer_graph.nodes[output_node]["buffer_descriptors"])
    buffer = model.buffer_descriptor_to_buffer[buffer_descriptor]
    shape = first(model.buffer_graph.nodes[output_node]["shapes"])
    return np.reshape(buffer.array[: math.prod(shape)], shape)


def evaluate_mosaic_model_with_kernel_fusion(model: ModelWithKernelFusion, output_var, inputs):
    initialize_variable_buffers(model.buffer_graph, inputs, model.buffer_descriptor_to_buffer)

    buffers = [model.buffer_descriptor_to_buffer[key] for key in model.buffer_descriptor_to_buffer]
    pointers = [buffer.data() for buffer in buffers]
    model.run_model(*pointers)

    output_node = first(model.buffer_graph.successors(output_var.node))
    buffer_descriptor = first(model.buffer_graph.nodes[output_node]["buffer_descriptors"])
    buffer = model.buffer_descriptor_to_buffer[buffer_descriptor]
    shape = first(model.buffer_graph.nodes[output_node]["shapes"])
    return np.reshape(buffer.array[: math.prod(shape)], shape)


def evaluate_mosaic_model(model: ModelWithoutKernelFusion | ModelWithKernelFusion, output_var, inputs):
    if isinstance(model, ModelWithKernelFusion):
        return evaluate_mosaic_model_with_kernel_fusion(model, output_var, inputs)
    return evaluate_mosaic_model_without_kernel_fusion(model, output_var, inputs)
