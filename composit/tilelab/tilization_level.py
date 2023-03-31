from __future__ import annotations

from pyrsistent import PClass, field


class TilizationLevel(PClass):
    level_name = field()
    tile_shape = field()
