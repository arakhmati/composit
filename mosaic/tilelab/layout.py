from pyrsistent import PClass, field


class DefaultLayout(PClass):
    ...


class TransposedLayout(PClass):
    order: tuple[int, ...] = field()
