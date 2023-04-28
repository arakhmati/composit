import collections

from mosaic.tilelab.tile import ArrayTileConfig
from mosaic.tilelab.tile_view import TileLevel


def normalize_value(value):
    if isinstance(value, str):
        return value
    elif isinstance(value, ArrayTileConfig):
        return f"{normalize_value(value.shape)}_{normalize_value(value.hierarchy)}_{normalize_value(value.layout)}"
    elif isinstance(value, TileLevel):
        return f"{value.level_name}_{normalize_value(value.tile_shape)}"
    elif isinstance(value, collections.abc.Iterable):
        return "_".join([normalize_value(element) for element in value])
    return str(value)


def create_kernel_name(*args):
    kernel_name = "__".join(f"{normalize_value(value)}" for value in args)
    for character in [" ", "-", "(", ")", "=", ","]:
        kernel_name = kernel_name.replace(character, "_")
    return kernel_name


__all__ = ["create_kernel_name"]
