from collections import deque

import numpy as np
import pytest

import transformers
from loguru import logger
from pyrsistent import pmap

import composit as cnp
from composit.multidigraph import compose_all, topological_traversal, visualize_graph
from composit.nn import Variable
from composit.numpy.core import Constant

from model_zoo.bert import (
    create_bert_config,
    functional_bert,
    convert_parameters_to_numpy,
)


class Buffer:
    global_index = 0

    def __init__(self):
        self.index = Buffer.global_index
        Buffer.global_index += 1
        self.current = None

    def __repr__(self):
        return f"Buffer(id={self.index})"


def try_returning_buffer_to_queue(graph, node, node_to_users, node_to_buffer, buffers_queue):
    for predecessor, _ in graph.in_edges(node):
        if isinstance(graph.nodes[predecessor]["instruction"], (Constant, Variable)):
            continue

        if predecessor not in node_to_users:
            continue

        node_to_users = node_to_users.set(predecessor, node_to_users[predecessor] - 1)
        if node_to_users[predecessor] == 0 and predecessor in node_to_buffer:
            buffer = node_to_buffer[predecessor]
            if buffer.current == predecessor:
                buffers_queue.append(buffer)

    return node_to_users


def can_instruction_reuse_buffer(instruction):
    return type(instruction).__name__ not in {"matmul"}


def propagate_buffer_down(graph, node, node_to_users, node_to_buffer):
    if graph.out_degree(node) != 1:
        return node_to_users, node_to_buffer

    (successor,) = graph.successors(node)

    successor_instruction = graph.nodes[successor]["instruction"]
    if not can_instruction_reuse_buffer(successor_instruction):
        return node_to_users, node_to_buffer

    if successor in node_to_buffer:
        return node_to_users, node_to_buffer

    buffer = node_to_buffer[node]
    node_to_buffer = node_to_buffer.set(successor, buffer)
    buffer.current = successor

    return propagate_buffer_down(graph, successor, node_to_users, node_to_buffer)


def analyze_buffer_reuse(*outputs):
    graph = compose_all(*(output.graph for output in outputs))

    buffers_queue = deque(maxlen=None)

    node_to_users = pmap()
    node_to_buffer = pmap()

    for node in graph:
        node_to_users = node_to_users.set(node, graph.out_degree(node))

    for node in topological_traversal(graph):
        if node in node_to_buffer:
            node_to_users = try_returning_buffer_to_queue(graph, node, node_to_users, node_to_buffer, buffers_queue)
            continue

        instruction = graph.nodes[node]["instruction"]
        if isinstance(instruction, (Constant, Variable)):
            continue

        if not buffers_queue:
            buffer = Buffer()
        else:
            buffer = buffers_queue.popleft()

        node_to_buffer = node_to_buffer.set(node, buffer)
        buffer.current = node
        node_to_users, node_to_buffer = propagate_buffer_down(graph, node, node_to_users, node_to_buffer)
        node_to_users = try_returning_buffer_to_queue(graph, node, node_to_users, node_to_buffer, buffers_queue)

    buffers_set = {buffer for buffer in node_to_buffer.values()}
    logger.info(len(buffers_set))

    def visualize_node(graphviz_graph, graph, node):
        colors = [
            "red",
            "blue",
            "green",
            "yellow",
            "purple",
            "orange",
        ]

        if node in node_to_buffer:
            buffer = node_to_buffer[node]
            color = colors[buffer.index % len(colors)]
            style = "filled"
        else:
            color = "black"
            style = "filled"
        fontcolor = {"yellow": "black"}.get(color, "white")
        graphviz_graph.node(node.name, label=f"{node}", color=color, style=style, fontcolor=fontcolor)

    visualize_graph(graph, visualize_node=visualize_node, render=False)


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

    analyze_buffer_reuse(model)
