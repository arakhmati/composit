from ctypes import cdll

from loguru import logger
from pyrsistent import pmap
from toolz import first

import codegen as c
from composit.introspection import class_name
from composit.multidigraph import topological_traversal
from composit.numpy.core import get_operands
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
from mosaic.backends.x86.types import BufferType


def get_kernel_name_and_module(instruction, input_tile_configs, output_tile_config, input_dtypes, output_dtype):
    instruction_class_name = class_name(instruction)

    kernel_name = None
    kernel_module = None
    if instruction_class_name in {"Constant", "Variable"}:
        pass
    elif instruction_class_name == "Tilize":
        kernel_name, kernel_module = tilize.generate_module(
            input_tile_configs,
            output_tile_config,
            input_dtypes,
            output_dtype,
        )
    elif instruction_class_name == "Untilize":
        kernel_name, kernel_module = untilize.generate_module(
            input_tile_configs,
            output_tile_config,
            input_dtypes,
            output_dtype,
        )
    elif instruction_class_name == "matmul":
        kernel_name, kernel_module = matrix_multiplication.generate_module(
            input_tile_configs,
            output_tile_config,
            input_dtypes,
            output_dtype,
            use_avx_manually=True,
        )
    elif instruction_class_name in {"exp", "sqrt", "gelu"}:
        kernel_name, kernel_module = unary_operation.generate_module(
            input_tile_configs,
            output_tile_config,
            input_dtypes,
            output_dtype,
            instruction_class_name,
        )
    elif instruction_class_name in {"add", "subtract", "divide", "multiply"}:
        kernel_name, kernel_module = binary_operation.generate_module(
            input_tile_configs, output_tile_config, input_dtypes, output_dtype, instruction_class_name
        )
    elif instruction_class_name in {"reshape"}:
        pass
    elif instruction_class_name in {"sum", "mean", "max"}:
        kernel_name, kernel_module = reduce.generate_module(
            input_tile_configs, output_tile_config, input_dtypes, output_dtype, instruction_class_name
        )
    elif instruction_class_name in {"embedding"}:
        kernel_name, kernel_module = embedding.generate_module(
            input_tile_configs, output_tile_config, input_dtypes, output_dtype
        )
    elif instruction_class_name in {"transpose"}:
        kernel_name, kernel_module = transpose.generate_module(
            input_tile_configs, output_tile_config, input_dtypes, output_dtype, instruction.axes
        )
    else:
        raise NotImplementedError(f"There is no kernel implementation for {instruction_class_name}")
    return kernel_name, kernel_module


def generate_and_compile_kernels(graph, test_output_path, node_output_to_tile_config):
    node_to_kernel_name = {}
    kernel_name_to_kernel_module = {}

    for node, attributes in graph.nodes(data=True):
        instruction = attributes["instruction"]

        input_tile_configs = [
            node_output_to_tile_config[(input_node, output_index)]
            for input_node, output_index in get_operands(graph, node)
        ]
        output_tile_config = node_output_to_tile_config[(node, 0)]

        input_dtypes = [
            graph.nodes[input_node]["dtypes"][output_index] for input_node, output_index in get_operands(graph, node)
        ]
        output_dtype = attributes["dtypes"][0]

        kernel_name, kernel_module = get_kernel_name_and_module(
            instruction, input_tile_configs, output_tile_config, input_dtypes, output_dtype
        )

        node_to_kernel_name[node] = kernel_name
        if kernel_module is not None:
            kernel_name_to_kernel_module[kernel_name] = kernel_module

    kernel_name_to_run_kernel = {}
    for kernel_name in set(node_to_kernel_name.values()):
        if kernel_name is None:
            kernel_name_to_run_kernel[kernel_name] = lambda *_: None
        else:
            source_file_name = (test_output_path / kernel_name).with_suffix(".cpp")
            kernel_module = kernel_name_to_kernel_module[kernel_name]
            kernel_module.save(source_file_name)

            shared_library_file = compile_shared_library(source_file_name)
            shared_library = cdll.LoadLibrary(shared_library_file)
            run_kernel = getattr(shared_library, kernel_name)
            kernel_name_to_run_kernel[kernel_name] = run_kernel

    node_to_run_kernel = {
        node: kernel_name_to_run_kernel[kernel_name] for node, kernel_name in node_to_kernel_name.items()
    }
    return pmap(node_to_run_kernel)


def generate_and_compile_run_model(graph, test_output_path, node_output_to_tile_config, buffer_descriptor_to_buffer):
    node_to_kernel_name = {}
    module = c.Module(includes=[], members=[])
    kernel_names_in_module = set()

    for node, attributes in graph.nodes(data=True):
        instruction = attributes["instruction"]

        input_tile_configs = [
            node_output_to_tile_config[(input_node, output_index)]
            for input_node, output_index in get_operands(graph, node)
        ]
        output_tile_config = node_output_to_tile_config[(node, 0)]

        input_dtypes = [
            graph.nodes[input_node]["dtypes"][output_index] for input_node, output_index in get_operands(graph, node)
        ]
        output_dtype = attributes["dtypes"][0]

        kernel_name, kernel_module = get_kernel_name_and_module(
            instruction, input_tile_configs, output_tile_config, input_dtypes, output_dtype
        )

        node_to_kernel_name[node] = kernel_name
        if kernel_name not in kernel_names_in_module and kernel_name is not None:
            module += kernel_module
            kernel_names_in_module.add(kernel_name)

    arguments = []
    buffer_descriptor_to_variable = {}
    for buffer_descriptor in sorted(buffer_descriptor_to_buffer):
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
        kernel_name = node_to_kernel_name[node]
        if kernel_name is None:
            continue

        input_vars = [
            buffer_descriptor_to_variable[first(graph.nodes[input_node]["buffer_descriptors"])]
            for input_node, _ in get_operands(graph, node)
        ]
        output_var = buffer_descriptor_to_variable[first(graph.nodes[node]["buffer_descriptors"])]
        invocation = c.invoke(kernel_name, *input_vars, output_var)
        statements.append(c.Statement(invocation))

    model_name = "run_model"
    module += c.Module(
        includes=[],
        members=[
            c.Function(
                return_type=c.Type("void"),
                name=c.Identifier(model_name),
                arguments=arguments,
                body=c.block(*statements),
            ).extern_c()
        ],
    )

    model_source_file = (test_output_path / model_name).with_suffix(".cpp")
    module.save(model_source_file)

    shared_library_file = compile_shared_library(model_source_file)
    shared_library = cdll.LoadLibrary(shared_library_file)
    run_model = getattr(shared_library, model_name)
    return run_model
