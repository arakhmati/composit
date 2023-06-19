import numpy as np

import composit as cnp


def test_matmul(m=5, k=10, n=3):
    def function(input_var):
        weights = cnp.random.random((k, n))
        bias = cnp.random.random((n,))
        output_var = cnp.matmul(input_var, weights, dtype="int8")
        output_var += bias
        return output_var

    input_var = cnp.asarray(np.random.random((m, k)), name="input_var")
    output_var = function(input_var)
    first_hash = hash(output_var)
    assert first_hash == hash(function(input_var))

    input_var = cnp.ndarray((m, k), name="input_var")
    output_var = function(input_var)
    assert first_hash == hash(output_var)

    input_var = cnp.asarray(np.zeros((m, k)), name="input_var")
    output_var = function(input_var)
    assert first_hash == hash(output_var)
