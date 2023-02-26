from pyimmer import pmap


def test_pmap_initialization_with_empty_dict():
    m = pmap({})
    assert len(m) == 0


def test_pmap_initialization_with_one_key_value_pair():
    key, value = "key", "value"
    m = pmap({key: value})
    assert len(m) == 1
    assert value == m.get(key)


def test_pmap_initialization_with_two_key_value_pairs():
    key_0, value_0 = "key", "value"
    key_1, value_1 = 123, 456
    m = pmap({key_0: value_0, key_1: value_1})
    assert len(m) == 2
    assert value_0 == m.get(key_0)
    assert value_1 == m.get(key_1)
