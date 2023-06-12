from typing import Tuple

from pyrsistent import PClass, field


class DefaultLayout(PClass):
    ...


class TransposedLayout(PClass):
    order: Tuple[int, ...] = field()
