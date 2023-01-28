import collections
import inspect

import graphviz
import numpy as np
from pyrsistent import PClass, field

import persistent_numpy as pnp
from persistent_numpy.persistent_array import PersistentArray, Node
from persistent_numpy.multidigraph import MultiDiGraph, compose_all, visualize_graph


class ModuleArgument(PClass):
    def __call__(self, *args):
        return args


def module_argument(*, name: str, shape: tuple) -> PersistentArray:
    node = Node(name=name)
    graph = MultiDiGraph().add_node(node, instruction=ModuleArgument(), shapes=(shape,))
    return PersistentArray(graph=graph, node=node)


class Module(PClass):
    module_function = field()
    graph: MultiDiGraph = field()
    input_vars = field()
    output_vars = field()

    def __call__(self, *input_arrays: list[np.ndarray]):
        inputs = {input_var: input_array for input_var, input_array in zip(self.input_vars, input_arrays)}
        output = pnp.nn.evaluate(*self.output_vars, inputs=inputs)
        return output


def create_module(module_function, input_vars, module_input_vars, output_vars):
    module_graph = compose_all(*tuple(output_var.graph for output_var in output_vars))
    return pnp.create_from_numpy_compute_instruction(
        *input_vars,
        instruction=Module(
            module_function=module_function, graph=module_graph, input_vars=module_input_vars, output_vars=output_vars
        ),
    )


def wrap_module(module_function):
    module_function_parameters = inspect.signature(module_function).parameters

    def wrapper(*input_vars, **kwargs):
        kwargs.update(
            {key: value.default for key, value in module_function_parameters.items() if value.default != inspect._empty}
        )

        # Convert input vars to nn.variables
        module_input_vars = tuple(
            module_argument(name=f"{input_var.node.name}_{input_var.output_index}", shape=input_var.shape)
            for input_var in input_vars
        )
        output_vars = module_function(*module_input_vars, **kwargs)
        if not isinstance(output_vars, collections.abc.Iterable):
            output_vars = (output_vars,)

        return create_module(module_function, input_vars, module_input_vars, output_vars)

    return wrapper


LEVEL_COLORS = [
    "blue",
    "red",
    "green",
    "yellow",
    "purple",
]


def visualize_modules(
    graph: MultiDiGraph,
    *,
    graphviz_graph=None,
    level=0,
    level_prefix="",
    render=False,
) -> None:

    if graphviz_graph is None:
        graphviz_graph = graphviz.Digraph()

    def visualize_node(graphviz_graph, graph, node):
        instruction = graph.nodes[node]["instruction"]
        name = f"{level_prefix}_{node.name}_{level}"
        if isinstance(instruction, Module):
            with graphviz_graph.subgraph(name=node.name) as cluster_graph:
                cluster_graph.attr(
                    color=LEVEL_COLORS[level % len(LEVEL_COLORS)],
                    cluster="true",
                    label=instruction.module_function.__name__,
                )
                visualize_modules(instruction.graph, graphviz_graph=cluster_graph, level=level + 1, level_prefix=name)
        else:
            graphviz_graph.node(name, label=f"{type(instruction).__name__}")

    def visualize_edge(graphviz_graph, graph, edge):
        source, sink, keys, data = edge

        def get_source_name(graph, node, level, prefix):
            name = f"{prefix}_{node.name}_{level}"
            instruction = graph.nodes[node]["instruction"]
            if not isinstance(instruction, Module):
                return name
            module = instruction
            module_graph = module.graph
            module_node = module.output_vars[data["source_output_index"]].node
            return get_source_name(module_graph, module_node, level + 1, prefix=name)

        def get_sink_name(graph, node, level, prefix):
            name = f"{prefix}_{node.name}_{level}"
            instruction = graph.nodes[node]["instruction"]
            if not isinstance(instruction, Module):
                return name
            module = instruction
            module_graph = module.graph
            module_node = module.input_vars[data["sink_input_index"]].node
            return get_sink_name(module_graph, module_node, level + 1, prefix=name)

        source_name = get_source_name(graph, source, level, prefix=level_prefix)
        sink_name = get_sink_name(graph, sink, level, prefix=level_prefix)

        if source_name == sink_name:
            return

        graphviz_graph.edge(
            source_name,
            sink_name,
            label=f"{data['source_output_index']} - {data['sink_input_index']}",
        )

    visualize_graph(
        graph,
        graphviz_graph=graphviz_graph,
        visualize_node=visualize_node,
        visualize_edge=visualize_edge,
        render=render and level == 0,
    )


__all__ = ["Module", "wrap_module", "visualize_modules"]
