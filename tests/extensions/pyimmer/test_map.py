import pytest

from pyimmer import pmap


def test_initialize_empty_pmap():
    m = pmap()
    assert len(m) == 0


def test_initialize_empty_pmap_from_dict():
    m = pmap({})
    assert len(m) == 0


def test_initialize_pmap_from_dict_and_get_first_item():
    key, value = "key", "value"
    m = pmap({key: value})
    assert len(m) == 1
    assert value == m.get(key)
    assert value == m[key]


def test_initialize_pmap_from_dict_and_get_first_and_second_items():
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


def test_initialize_empty_pmap_and_set_item():
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


def test_initialize_empty_pmap_and_iterate_over_it():
    m = pmap()
    assert len(m) == 0
    for _ in m:
        ...


def test_initialize_pmap_with_one_item_and_iterate_over_it():
    key, value = "key", "value"
    m = pmap({key: value})
    assert len(m) == 1
    for iterator_key in m:
        assert key == iterator_key


def test_initialize_pmap_with_two_items_and_iterate_over_it():
    key_0, value_0 = "key", "value"
    key_1, value_1 = 123, 456
    m = pmap({key_0: value_0, key_1: value_1})

    assert len(m) == 2
    keys = [key_0, key_1]
    for iterator_key in m:
        assert iterator_key in keys
        keys.remove(iterator_key)


def test_initialize_pmap_with_three_items_and_iterate_over_it():
    key_0, value_0 = "key", "value"
    key_1, value_1 = 123, 456
    key_2, value_2 = 321, 654
    m = pmap({key_0: value_0, key_1: value_1, key_2: value_2})

    assert len(m) == 3
    keys = [key_0, key_1, key_2]
    for iterator_key in m:
        assert iterator_key in keys
        keys.remove(iterator_key)


def test_initialize_empty_pmap_and_iterate_over_keys():
    m = pmap()
    assert len(m) == 0
    for _ in m.keys():
        ...


def test_initialize_pmap_with_one_item_and_iterate_over_keys():
    key, value = "key", "value"
    m = pmap({key: value})
    assert len(m) == 1
    for iterator_key in m.keys():
        assert key == iterator_key


def test_initialize_pmap_with_two_items_and_iterate_over_keys():
    key_0, value_0 = "key", "value"
    key_1, value_1 = 123, 456
    m = pmap({key_0: value_0, key_1: value_1})

    assert len(m) == 2
    keys = [key_0, key_1]
    for iterator_key in m.keys():
        assert iterator_key in keys
        keys.remove(iterator_key)
    assert len(keys) == 0


def test_initialize_pmap_with_three_items_and_iterate_over_keys():
    key_0, value_0 = "key", "value"
    key_1, value_1 = 123, 456
    key_2, value_2 = 321, 654
    m = pmap({key_0: value_0, key_1: value_1, key_2: value_2})

    assert len(m) == 3
    keys = [key_0, key_1, key_2]
    for iterator_key in m.keys():
        assert iterator_key in keys
        keys.remove(iterator_key)
    assert len(keys) == 0


def test_initialize_empty_pmap_and_iterate_over_values():
    m = pmap()
    assert len(m) == 0
    for _ in m.values():
        ...


def test_initialize_pmap_with_one_item_and_iterate_over_values():
    key, value = "key", "value"
    m = pmap({key: value})
    assert len(m) == 1
    for iterator_value in m.values():
        assert value == iterator_value


def test_initialize_pmap_with_two_items_and_iterate_over_values():
    key_0, value_0 = "key", "value"
    key_1, value_1 = 123, 456
    m = pmap({key_0: value_0, key_1: value_1})

    assert len(m) == 2
    values = [value_0, value_1]
    for iterator_value in m.values():
        assert iterator_value in values
        values.remove(iterator_value)
    assert len(values) == 0


def test_initialize_pmap_with_three_items_and_iterate_over_values():
    key_0, value_0 = "key", "value"
    key_1, value_1 = 123, 456
    key_2, value_2 = 321, 654
    m = pmap({key_0: value_0, key_1: value_1, key_2: value_2})

    assert len(m) == 3
    values = [value_0, value_1, value_2]
    for iterator_value in m.values():
        assert iterator_value in values
        values.remove(iterator_value)
    assert len(values) == 0


def test_initialize_empty_pmap_and_iterate_over_items():
    m = pmap()
    assert len(m) == 0
    for _ in m.items():
        ...


def test_initialize_pmap_with_one_item_and_iterate_over_items():
    key, value = "key", "value"
    m = pmap({key: value})
    assert len(m) == 1
    for iterator_key, iterator_value in m.items():
        assert key == iterator_key
        assert value == iterator_value


def test_initialize_pmap_with_two_items_and_iterate_over_items():
    key_0, value_0 = "key", "value"
    key_1, value_1 = 123, 456
    m = pmap({key_0: value_0, key_1: value_1})

    assert len(m) == 2
    items = [(key_0, value_0), (key_1, value_1)]
    for iterator_item in m.items():
        assert iterator_item in items
        items.remove(iterator_item)
    assert len(items) == 0


def test_initialize_pmap_with_three_items_and_iterate_over_items():
    key_0, value_0 = "key", "value"
    key_1, value_1 = 123, 456
    key_2, value_2 = 321, 654
    m = pmap({key_0: value_0, key_1: value_1, key_2: value_2})

    assert len(m) == 3
    items = [(key_0, value_0), (key_1, value_1), (key_2, value_2)]
    for iterator_item in m.items():
        assert iterator_item in items
        items.remove(iterator_item)
    assert len(items) == 0
