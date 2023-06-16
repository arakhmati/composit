import numpy as np
import torch

import composit as cnp
import flashlight


# Remove this global state variable by making it local to tracer
graph_input_index = 0


def reset_graph_input_index():
    global graph_input_index
    graph_input_index = 0


def convert_torch_tensors_to_lazy_tensors(*args, allow_scalars=False):
    global graph_input_index
    first_arg = None
    for index, arg in enumerate(args):
        if isinstance(arg, flashlight.Tensor):
            result = arg.lazy_tensor
        elif isinstance(arg, torch.Tensor):
            np_tensor = arg.detach().numpy()
            result = cnp.asarray(np_tensor, name=f"tensor_{graph_input_index:16}")
            graph_input_index += 1
        elif allow_scalars and isinstance(arg, (int, float)):
            np_tensor = np.asarray(arg, first_arg.dtype)
            result = cnp.asarray(np_tensor, name=f"scalar_{arg}")
        else:
            continue

        if index == 0:
            first_arg = result

        yield result
