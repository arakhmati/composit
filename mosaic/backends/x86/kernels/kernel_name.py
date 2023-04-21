import collections

from mosaic.tilelab.tile import ArrayTileConfig
from mosaic.tilelab.tile_view import TileLevel


def normalize_value(value):
    if isinstance(value, str):
        return value
    elif isinstance(value, ArrayTileConfig):
        return f"{normalize_value(value.shape)}_{normalize_value(value.hierarchy)}"
    elif isinstance(value, TileLevel):
        return f"{value.level_name}_{normalize_value(value.tile_shape)}"
    elif isinstance(value, collections.abc.Iterable):
        return "_".join([normalize_value(element) for element in value])
    return str(value)


def create_kernel_name(*args):
    return "__".join(f"{normalize_value(value)}" for value in args).replace(" ", "_")


__all__ = ["create_kernel_name"]
