import torch
from .linear import Linear
from .embedding import Embedding


class TiedLinear(Linear):
    def __init__(self, embedding: Embedding, bias: bool = True):
        super().__init__(embedding.embed.shape[1], embedding.embed.shape[0], bias=bias)
        del self._parameters["weights"]
        self.embedding = embedding

    @property
    def weights(self) -> torch.Tensor:
        return self.embedding.embed.t()
