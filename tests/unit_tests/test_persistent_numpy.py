import pytest

import numpy
import persistent_numpy


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


@pytest.mark.parametrize("np", [numpy, persistent_numpy])
def test_matmul(np):
    array_a = np.ones((5, 10))
    array_b = np.ones((10, 3))
    result = np.matmul(array_a, array_b)
    assert result.shape == (5, 3)


@pytest.mark.parametrize("np", [numpy, persistent_numpy])
def test_add(np):
    array_a = np.ones((5, 10))
    array_b = np.ones((10,))
    result = array_a + array_b
    assert result.shape == (5, 10)


@pytest.mark.parametrize("np", [numpy, persistent_numpy])
def test_add_scalar(np):
    array_a = np.ones((5, 10))
    array_b = 1
    result = array_a + array_b
    assert result.shape == (5, 10)


@pytest.mark.parametrize("np", [numpy, persistent_numpy])
def test_subtract(np):
    array_a = np.ones((5, 10))
    array_b = np.ones((10,))
    result = array_a - array_b
    assert result.shape == (5, 10)


@pytest.mark.parametrize("np", [numpy, persistent_numpy])
def test_multiply(np):
    array_a = np.ones((5, 10))
    array_b = np.ones((10,))
    result = array_a * array_b
    assert result.shape == (5, 10)


@pytest.mark.parametrize("np", [numpy, persistent_numpy])
def test_divide(np):
    array_a = np.ones((5, 10))
    array_b = np.ones((10,))
    result = array_a / array_b
    assert result.shape == (5, 10)


@pytest.mark.parametrize("np", [numpy, persistent_numpy])
def test_transpose(np):
    array = np.ones((5, 25, 15))
    result = np.transpose(array, axes=(2, 0, 1))
    assert result.shape == (15, 5, 25)


@pytest.mark.parametrize("np", [numpy, persistent_numpy])
def test_exp(np):
    array = np.ones((5, 25, 15))
    result = np.exp(array)
    assert result.shape == (5, 25, 15)


@pytest.mark.parametrize("np", [numpy, persistent_numpy])
def test_sum(np):
    array = np.ones((5, 25, 15))
    result = np.sum(array, axis=-1)
    assert result.shape == (5, 25)
