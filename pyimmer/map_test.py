import pytest

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
    assert value == m[key]


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


def test_initialize_empty_pmap_and_update_it_from_dict():
    m = pmap()
    assert len(m) == 0

    key_0, value_0 = "key", "value"
    key_1, value_1 = 123, 456
    new_m = m.update({key_0: value_0, key_1: value_1})

    assert len(m) == 0
    assert len(new_m) == 2
    assert value_0 == new_m.get(key_0)
    assert value_1 == new_m.get(key_1)


def test_initialize_empty_pmap_and_update_it_from_pmap():
    m = pmap()
    assert len(m) == 0

    key_0, value_0 = "key", "value"
    key_1, value_1 = 123, 456
    new_m = m.update(pmap({key_0: value_0, key_1: value_1}))

    assert len(m) == 0
    assert len(new_m) == 2
    assert value_0 == new_m.get(key_0)
    assert value_1 == new_m.get(key_1)


def test_initialize_empty_pmap_and_set_key_value_pair():
    m = pmap()
    assert len(m) == 0

    key, value = "key", "value"
    new_m = m.set(key, value)

    assert len(m) == 0
    assert len(new_m) == 1
    assert value == new_m.get(key)


def test_initialize_empty_pmap_from_dict_and_check_that_it_does_not_contain_key():
    key = "key"
    m = pmap()
    assert len(m) == 0
    assert key not in m
    with pytest.raises(KeyError):
        assert m[key]


def test_initialize_pmap_from_dict_and_check_that_it_contains_key():
    key, value = "key", "value"
    m = pmap({key: value})
    assert len(m) == 1
    assert key in m
