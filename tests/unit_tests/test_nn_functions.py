import numpy as np
import torch.nn.functional

import persistent_numpy as pnp
import persistent_numpy.nn_functions as pnn


def test_gelu():
    array = pnp.ones((5, 25, 15))
    result = pnn.gelu(array)

    torch_array = torch.from_numpy(pnp.to_numpy(array)).float()
    torch_result = torch.nn.functional.gelu(torch_array).numpy()

    assert result.shape == torch_result.shape
    assert np.allclose(pnp.to_numpy(result), torch_result)
