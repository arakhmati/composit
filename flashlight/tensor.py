import torch

import composit as cnp


class Tensor(torch.Tensor):
    @staticmethod
    def __new__(cls, tensor, lazy_tensor, *args, **kwargs):
        return super().__new__(cls, tensor, *args, **kwargs)

    def __init__(self, _, lazy_tensor):
        super().__init__()
        self.lazy_tensor = lazy_tensor

    @property
    def graph(self):
        return self.lazy_tensor.graph

    def __repr__(self):
        return f"flashlight.{super().__repr__()}, var={self.lazy_tensor}"

def forward(*output_tensors, input_tensors):
    return cnp.nn.evaluate(
        *(output_tensor.lazy_tensor for output_tensor in output_tensors),
        inputs={
            input_tensor.lazy_tensor: input_tensor.numpy()
            for input_tensor in input_tensors
        }
    )
