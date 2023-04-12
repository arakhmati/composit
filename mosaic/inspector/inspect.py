import math

import numpy as np
from loguru import logger
import pandas as pd

from composit.multidigraph import compose_all, topological_traversal

pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", 0)


def format_bytes(size):
    power = 2**10
    n = 0
    power_labels = {0: "", 1: "KB", 2: "MB", 3: "GB", 4: "TB"}
    while size > power:
        size /= power
        n += 1
    return f"{size:3.3f} {power_labels[n]}"


def inspect(*outputs):
    graph = compose_all(*(output.graph for output in outputs))

    dataframe = pd.DataFrame(columns=["name", "type_name", "shape", "dtype", "volume", "memory_in_bytes"])

    for node in topological_traversal(graph):
        attributes = graph.nodes[node]

        name = node.name

        instruction = attributes["instruction"]
        type_name = type(instruction).__name__

        shapes = attributes["shapes"]
        assert len(shapes) == 1
        shape = shapes[0]

        dtypes = attributes["dtypes"]
        assert len(dtypes) == 1
        dtype = dtypes[0]

        volume = math.prod(shape)
        memory_in_bytes = volume * np.dtype(dtype).itemsize

        dataframe.loc[len(dataframe)] = [
            name,
            type_name,
            shape,
            dtype,
            volume,
            memory_in_bytes,
        ]

    logger.info(f"\n{dataframe}")

    total_memory_used_without_reuse = dataframe["memory_in_bytes"].sum()
    logger.info(f"total_memory_used_without_reuse = {format_bytes(total_memory_used_without_reuse)}")

    unique_ops = (
        dataframe[["shape", "type_name", "dtype"]].drop_duplicates().reset_index(drop=True).sort_values("type_name")
    )
    logger.info(f"\n{unique_ops}")
