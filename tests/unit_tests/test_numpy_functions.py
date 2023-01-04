import pytest

import numpy

import persistent_numpy


def check_results(function):
    np_result = function(numpy)
    pnp_result = function(persistent_numpy)

    assert np_result.shape == pnp_result.shape
    assert numpy.allclose(np_result, pnp_result.to_numpy())


@pytest.mark.parametrize("np", [numpy, persistent_numpy])
def test_ndarray(np):
    array = np.ndarray((5, 10))
    assert array.shape == (5, 10)


@pytest.mark.parametrize("np", [numpy, persistent_numpy])
def test_named_ndarray(np):
    if not hasattr(np, "named_ndarray"):
        pytest.skip("current np library doesn't have named_ndarray")
    array = np.named_ndarray((5, 10), name="activations")
    assert array.shape == (5, 10)
    assert len(array.graph) == 1


@pytest.mark.parametrize("np", [numpy, persistent_numpy])
def test_zeros(np):
    array = np.zeros((5, 10))
    assert array.shape == (5, 10)


@pytest.mark.parametrize("np", [numpy, persistent_numpy])
def test_ones(np):
    array = np.ones((5, 10))
    assert array.shape == (5, 10)


def test_matmul():
    def function(np):
        array_a = np.ones((5, 10))
        array_b = np.ones((10, 3))
        result = np.matmul(array_a, array_b)
        return result

    check_results(function)


def test_add():
    def function(np):
        array_a = np.ones((5, 10))
        array_b = np.ones((10,))
        result = array_a + array_b
        return result

    check_results(function)


def test_add_scalar():
    def function(np):
        array_a = np.ones((5, 10))
        array_b = 1
        result = array_a + array_b
        return result

    check_results(function)


def test_subtract():
    def function(np):
        array_a = np.ones((5, 10))
        array_b = np.ones((10,))
        result = array_a - array_b
        return result

    check_results(function)


def test_multiply():
    def function(np):
        array_a = np.ones((5, 10))
        array_b = np.ones((10,))
        result = array_a * array_b
        return result

    check_results(function)


def test_divide():
    def function(np):
        array_a = np.ones((5, 10))
        array_b = np.ones((10,))
        result = array_a / array_b
        return result

    check_results(function)


def test_reshape():
    def function(np):
        array = np.ones((5, 25, 15), dtype="int32")
        result = np.reshape(array, (125, 15))
        return result

    check_results(function)


def test_transpose_with_axes_as_kwarg():
    def function(np):
        array = np.ones((5, 25, 15), dtype="int32")
        result = np.transpose(array, axes=(2, 0, 1))
        return result

    check_results(function)


def test_transpose_with_axes_as_arg():
    def function(np):
        array = np.ones((5, 25, 15), dtype="int32")
        result = np.transpose(array, (2, 0, 1))
        return result

    check_results(function)


def test_exp():
    def function(np):
        array = np.ones((5, 25, 15))
        result = np.exp(array)
        return result

    check_results(function)


def test_sum():
    def function(np):
        array = np.ones((5, 25, 15))
        result = np.sum(array, axis=-1)
        return result

    check_results(function)
