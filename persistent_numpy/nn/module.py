import collections
import inspect

import graphviz
import numpy as np
from pyrsistent import PClass, field

import persistent_numpy as pnp
from persistent_numpy.numpy.core import create_from_numpy_compute_instruction
from persistent_numpy.persistent_array import PersistentArray, Node
from persistent_numpy.multidigraph import MultiDiGraph, compose_all, visualize_graph

DISABLE = False  # Temporary hack until Module is supported in chain_rule


class ModuleInput(PClass):
    def __call__(self, *args):
        return args


def module_input(*, name: str, shape: tuple) -> PersistentArray:
    node = Node(name=name)
    graph = MultiDiGraph().add_node(node, instruction=ModuleInput(), shapes=(shape,))
    return PersistentArray(graph=graph, node=node)


class ModuleOutput(PClass):
    def __call__(self, *args):
        return args


def module_output(input_var) -> PersistentArray:
    return create_from_numpy_compute_instruction(
        input_var,
        instruction=ModuleOutput(),
    )


class Module(PClass):
    function = field()
    input_vars = field()
    output_vars = field()

    @property
    def graph(self):
        return compose_all(*tuple(module_output_var.graph for module_output_var in self.output_vars))

    def __call__(self, *input_arrays: list[np.ndarray]):
        inputs = {input_var: input_array for input_var, input_array in zip(self.input_vars, input_arrays)}
        output = pnp.nn.evaluate(*self.output_vars, inputs=inputs)
        return output


def create_module(module_function, operands, module_input_vars, module_output_var):
    return create_from_numpy_compute_instruction(
        *operands,
        instruction=Module(function=module_function, input_vars=module_input_vars, output_vars=module_output_var),
    )


def flatten_vars(vars, graph=None):
    """Convert input vars to nn.variables"""

    flat_vars = []
    for var in vars:

        if var is None:
            continue

        elif isinstance(var, PersistentArray):
            flat_vars.append(var)

        elif isinstance(var, dict):
            for key in sorted(var):
                value = var[key]
                if graph is not None:
                    if value.node in graph:
                        flat_vars.append(value)
                else:
                    flat_vars.append(value)

        else:
            raise ValueError

    return tuple(flat_vars)


def convert_input_vars_to_module_input_vars(input_vars):
    """Convert input vars to nn.variables"""

    module_input_vars = []
    for input_var in input_vars:

        if input_var is None:
            module_input_var = None

        elif isinstance(input_var, PersistentArray):
            name = input_var.node.name
            if input_var.output_index > 0:
                name = f"{name}_{input_var.output_index}"
            module_input_var = module_input(name=name, shape=input_var.shape)

        elif isinstance(input_var, dict):
            old_dict = input_var
            new_dict = {}
            for key in sorted(old_dict):
                value = old_dict[key]
                name = value.node.name
                if value.output_index > 0:
                    name = f"{name}_{value.output_index}"
                new_dict[key] = module_input(name=name, shape=value.shape)
            module_input_var = new_dict

        else:
            raise ValueError(f"{type(input_var)}")

        module_input_vars.append(module_input_var)

    return tuple(module_input_vars)


def wrap_module(module_function):
    module_function_parameters = inspect.signature(module_function).parameters

    def wrapper(*operands, **kwargs):
        if DISABLE:
            return module_function(*operands, **kwargs)

        kwargs.update(
            {key: value.default for key, value in module_function_parameters.items() if value.default != inspect._empty}
        )

        module_input_vars = convert_input_vars_to_module_input_vars(operands)

        module_output_vars = module_function(*module_input_vars, **kwargs)
        if not isinstance(module_output_vars, collections.abc.Iterable):
            module_output_vars = (module_output_vars,)

        module_output_vars = tuple(module_output(module_output_var) for module_output_var in module_output_vars)
        module_graph = compose_all(*tuple(module_output_var.graph for module_output_var in module_output_vars))

        flattened_operands = flatten_vars(operands, module_graph)
        flattened_module_input_vars = flatten_vars(module_input_vars, module_graph)

        return create_module(module_function, flattened_operands, flattened_module_input_vars, module_output_vars)

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
                    fontcolor="white",
                    bgcolor=LEVEL_COLORS[level % len(LEVEL_COLORS)],
                    cluster="true",
                    label=instruction.function.__name__,
                )
                cluster_graph.node_attr["style"] = "filled"
                cluster_graph.node_attr["fillcolor"] = "white"
                visualize_modules(instruction.graph, graphviz_graph=cluster_graph, level=level + 1, level_prefix=name)
        else:
            shapes = graph.nodes[node]["shapes"]
            shapes = shapes[0] if len(shapes) == 1 else shapes
            graphviz_graph.node(name, label=f"{type(instruction).__name__}:{shapes}")

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
            fontcolor="black" if level == 0 else "white",
        )

    visualize_graph(
        graph,
        graphviz_graph=graphviz_graph,
        visualize_node=visualize_node,
        visualize_edge=visualize_edge,
        render=render and level == 0,
    )


__all__ = ["Module", "wrap_module", "visualize_modules"]
