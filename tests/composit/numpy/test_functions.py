import pytest

import numpy

import composit


def check_results(function):
    np_array = function(numpy)
    cnp_array = function(composit)

    assert np_array.shape == cnp_array.shape
    assert numpy.allclose(np_array, composit.evaluate(cnp_array))


@pytest.mark.parametrize("np", [numpy, composit])
def test_ndarray(np):
    array = np.ndarray((5, 10), dtype="uint8")
    assert array.shape == (5, 10)


@pytest.mark.parametrize("np", [numpy, composit])
def test_named_ndarray(np):
    if not hasattr(np, "named_ndarray"):
        pytest.skip("current np library doesn't have named_ndarray")
    array = np.named_ndarray((5, 10), name="activations")
    assert array.shape == (5, 10)
    assert len(array.graph) == 1


@pytest.mark.parametrize("np", [numpy, composit])
def test_zeros(np):
    array = np.zeros((5, 10), dtype="uint8")
    assert array.shape == (5, 10)


@pytest.mark.parametrize("np", [numpy, composit])
def test_ones(np):
    array = np.ones((5, 10), dtype="uint8")
    assert array.shape == (5, 10)


def test_matmul():
    def function(np):
        array_a = np.ones((5, 10), dtype="uint8")
        array_b = np.ones((10, 3), dtype="uint8")
        result = np.matmul(array_a, array_b, dtype="int8")
        assert result.dtype == "int8"
        return result

    check_results(function)


def test_add():
    def function(np):
        array_a = np.ones((5, 10), dtype="uint8")
        array_b = np.ones((10,), dtype="uint8")
        result = array_a + array_b
        assert result.dtype == "uint8"
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


def test_sin():
    def function(np):
        array = np.ones((5, 25, 15))
        result = np.sin(array)
        return result

    check_results(function)


def test_cos():
    def function(np):
        array = np.ones((5, 25, 15))
        result = np.cos(array)
        return result

    check_results(function)


def test_tanh():
    def function(np):
        array = np.ones((5, 25, 15))
        result = np.tanh(array)
        return result

    check_results(function)


def test_sum():
    def function(np):
        array = np.ones((5, 25, 15))
        result = np.sum(array, axis=-1)
        return result

    check_results(function)


def test_get_item():
    def function(np):
        array = np.ones((5, 25, 15))
        result = array[3, 10]
        return result

    check_results(function)


def test_set_item():
    def function(np):
        numpy.random.seed(0)
        array = np.random.random((5, 25, 15))
        smaller_array = np.random.random((23, 15))
        if np == numpy:
            array[0, 2:] = smaller_array
        else:
            # Make a new language that supports persistent __setitem__? ;)
            array = np.set_item(array, (0, slice(2, None)), smaller_array)
        return array

    check_results(function)


def test_split():
    def function(np):
        numpy.random.seed(0)
        array = np.random.random((4, 8))
        result = np.split(array, 2, axis=0)
        result = [np.exp(element) for element in result]
        return result

    np_list = function(numpy)
    cnp_list = function(composit)

    for np_array, cnp_array in zip(np_list, cnp_list):
        assert np_array.shape == cnp_array.shape
        assert numpy.allclose(np_array, composit.evaluate(cnp_array))

    for np_array, cnp_array in zip(np_list, composit.evaluate(*cnp_list)):
        assert np_array.shape == cnp_array.shape
        assert numpy.allclose(np_array, cnp_array)


def test_concatenate():
    def function(np):
        array_a = np.ones((5, 10))
        array_b = np.ones((5, 15))
        result = np.concatenate((array_a, array_b), axis=1)
        return result

    check_results(function)
