from composit.multidigraph import compose_all
from mosaic.backends.x86.passes.create_buffers import create_buffers
from mosaic.backends.x86.passes.generate_and_compile import generate_and_compile_run_model, generate_and_compile_kernels
from mosaic.backends.x86.passes.insert_tilize_and_untilize_instructions import insert_tilize_and_untilize_instructions
from mosaic.backends.x86.types import (
    ModelWithKernelFusion,
    ModelWithoutKernelFusion,
)
from mosaic.tilelab.tile import create_tile_config
from mosaic.tilelab.tile_view import propagate_tile_views


def propagate_tile_config(graph, input_var_to_scheme):
    tile_views = propagate_tile_views(graph, inputs=input_var_to_scheme)
    node_output_to_tile_config = {node_output: create_tile_config(tile_view) for node_output, tile_view in tile_views}
    return node_output_to_tile_config


def compile_to_mosaic_model(
    *output_vars,
    input_var_to_scheme,
    output_path,
    reuse_buffers: bool,
    fuse_kernels: bool,
):
    graph = compose_all(*tuple(output_var.graph for output_var in output_vars))
    node_output_to_tile_config = propagate_tile_config(graph, input_var_to_scheme)
    graph, node_output_to_tile_config = insert_tilize_and_untilize_instructions(graph, node_output_to_tile_config)
    buffer_graph, buffer_descriptor_to_buffer = create_buffers(graph, node_output_to_tile_config, reuse_buffers)

    if fuse_kernels:
        run_model = generate_and_compile_run_model(
            buffer_graph, output_path, node_output_to_tile_config, buffer_descriptor_to_buffer
        )
        return ModelWithKernelFusion(
            buffer_graph=buffer_graph,
            run_model=run_model,
            buffer_descriptor_to_buffer=buffer_descriptor_to_buffer,
        )

    node_to_run_kernel = generate_and_compile_kernels(buffer_graph, output_path, node_output_to_tile_config)
    return ModelWithoutKernelFusion(
        buffer_graph=buffer_graph,
        node_to_run_kernel=node_to_run_kernel,
        buffer_descriptor_to_buffer=buffer_descriptor_to_buffer,
    )
