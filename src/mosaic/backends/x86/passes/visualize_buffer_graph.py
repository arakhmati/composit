import toolz

from pyrsistent import pvector
from pyimmer import pmap
from toolz import first

from composit.multidigraph import visualize_graph
from mosaic.backends.x86.types import BufferDescriptor, ConstantBufferDescriptor


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
    nodes = list(iterate_buffer_descriptors_to_nodes(graph))

    import colorsys

    hsv_tuples = [(i * 1.0 / 100, 0.5, 0.5) for i in range(len(nodes))]
    colors = ["#%02x%02x%02x" % tuple(map(lambda hsv: int(hsv * 255), colorsys.hsv_to_rgb(*hsv))) for hsv in hsv_tuples]

    return {buffer_descriptor: (color, "filled") for i, (buffer_descriptor, color) in enumerate(zip(nodes, colors))}


def visualize_node(graphviz_graph, graph, node):
    buffer_descriptor_to_color_and_style = create_buffer_descriptor_to_color_and_style(graph)
    buffer_descriptor = first(graph.nodes[node]["buffer_descriptors"])
    if isinstance(buffer_descriptor, ConstantBufferDescriptor):
        color = "black"
        fontcolor = "white"
        style = "filled"
    elif isinstance(buffer_descriptor, BufferDescriptor):
        color, style = buffer_descriptor_to_color_and_style[buffer_descriptor]
        fontcolor = "white"
    else:
        raise ValueError("Unrecognized BufferDescriptor")
    graphviz_graph.node(
        node.name, label=f"{node} @ {buffer_descriptor.name}", color=color, style=style, fontcolor=fontcolor
    )


def visualize_buffer_graph(graph):
    visualize_graph(graph, visualize_node=visualize_node, timeout=5)
