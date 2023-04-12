import math

import pandas as pd

from composit.multidigraph import compose_all, topological_traversal

pd.set_option("display.max_rows", None)


def inspect(*outputs):
    graph = compose_all(*(output.graph for output in outputs))

    dataframe = pd.DataFrame(columns=["name", "type_name", "shape", "volume", "dtype"])

    for node in topological_traversal(graph):
        name = node.name

        attributes = graph.nodes[node]
        shapes = attributes["shapes"]
        assert len(shapes) == 1
        shape = shapes[0]

        dtypes = attributes["dtypes"]
        assert len(dtypes) == 1
        dtype = dtypes[0]

        instruction = attributes["instruction"]
        type_name = type(instruction).__name__

        dataframe.loc[len(dataframe)] = [
            name,
            type_name,
            shape,
            math.prod(shape),
            dtype,
        ]

    print(dataframe)
