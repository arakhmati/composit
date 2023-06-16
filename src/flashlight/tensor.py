import torch


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
