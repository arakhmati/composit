from pyimmer import pmap


def test_initialize_empty_pmap():
    m = pmap()
    assert len(m) == 0


def test_initialize_empty_pmap_from_dict():
    m = pmap({})
    assert len(m) == 0


def test_initialize_pmap_from_dict_and_get_first_key_value_pair():
    key, value = "key", "value"
    m = pmap({key: value})
    assert len(m) == 1
    assert value == m.get(key)


def test_initialize_pmap_from_dict_and_get_first_and_second_key_value_pairs():
    key_0, value_0 = "key", "value"
    key_1, value_1 = 123, 456
    m = pmap({key_0: value_0, key_1: value_1})
    assert len(m) == 2
    assert value_0 == m.get(key_0)
    assert value_1 == m.get(key_1)


def test_initialize_pmap_from_dict_and_get_non_existent_key_without_default_value():
    key_0, value_0 = "key", "value"
    key_1, value_1 = 123, 456
    m = pmap({key_0: value_0, key_1: value_1})
    assert len(m) == 2
    assert m.get("non-existent key") is None


def test_initialize_pmap_from_dict_and_get_non_existent_key_with_default_value():
    key_0, value_0 = "key", "value"
    key_1, value_1 = 123, 456
    m = pmap({key_0: value_0, key_1: value_1})
    assert len(m) == 2

    default_value = "default value"
    assert default_value == m.get("non-existent key", default_value)

