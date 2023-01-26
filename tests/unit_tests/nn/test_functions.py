import numpy as np
import torch.nn.functional

import persistent_numpy as pnp


def test_embedding():
    input_tensor = pnp.zeros((5, 25), dtype=np.int32)
    weights = pnp.random.random((25, 10))
    result = pnp.nn.embedding(input_tensor, weights)

    torch_input_tensor = torch.from_numpy(pnp.to_numpy(input_tensor))
    torch_weights = torch.from_numpy(pnp.to_numpy(weights))
    torch_result = torch.nn.functional.embedding(torch_input_tensor, torch_weights).numpy()

    assert result.shape == torch_result.shape
    assert np.allclose(pnp.to_numpy(result), torch_result)


def test_gelu():
    array = pnp.random.random((5, 25, 15))
    result = pnp.nn.gelu(array)

    torch_array = torch.from_numpy(pnp.to_numpy(array))
    torch_result = torch.nn.functional.gelu(torch_array).numpy()

    assert result.shape == torch_result.shape
    assert np.allclose(pnp.to_numpy(result), torch_result)
