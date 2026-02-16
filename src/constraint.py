from abc import ABC, abstractmethod
from typing import Tuple, Union

import torch

from src.utils import (generate_C, generate_hypergraph,
                        get_H, )


class BaseConstraintGenerator(ABC):

    def __init__(self, m: int, n: int, beta: float = 2.):
        self.m = m
        self.n = n
        self.beta = beta
        self.X = (torch.rand(n, 1) > 0.5).float()
        self.is_generated = False 

    @abstractmethod
    def generate(self, **kwargs) -> Tuple[torch.Tensor, ...]:
        pass

    @abstractmethod
    def calculate_slack(self, solution_samples: torch.Tensor) -> torch.Tensor:
        pass

    def get_X(self) -> torch.Tensor:
        return self.X

    def to(self, device: torch.device):
        """Move internal tensors to given device."""
        self.X = self.X.to(device)
        # subclasses may have additional tensors; they should override or rely on this
        return self


class LinearConstraintGenerator(BaseConstraintGenerator):


    def __init__(self, m: int, n: int, beta: float = 2.):
        super().__init__(m, n, beta)
        self.A: Union[torch.Tensor, None] = None
        self.b: Union[torch.Tensor, None] = None

    def generate(
            self,
            num_items_per_constraint: int,
            max_abs_value: float = 1.) -> Tuple[torch.Tensor, torch.Tensor]:

        m, n = self.m, self.n

        if num_items_per_constraint <= 0 or num_items_per_constraint > n:
            raise ValueError("non_zero_per_row must be greater than 0 and not greater than n.")

        num_non_zeros = m * num_items_per_constraint

        row_indices = torch.arange(m).repeat_interleave(
            num_items_per_constraint)

        col_indices = []
        for i in range(m):
            cols = torch.randperm(n)[:num_items_per_constraint]
            col_indices.append(cols)
        col_indices = torch.cat(col_indices)
        indices = torch.stack([row_indices, col_indices])

        values = (torch.rand(num_non_zeros, dtype=torch.float32) * 2 *
                  max_abs_value) - max_abs_value


        A_sparse = torch.sparse_coo_tensor(indices,
                                           values, (m, n),
                                           dtype=torch.float32)
        A_dense = A_sparse.to_dense()


        AX = A_dense @ self.X

        # b} = AX} + \text{beta
        b = AX + self.beta

        self.A = A_dense
        self.b = b
        self.is_generated = True
        # print(self.A)


        return self.A, self.b

    def calculate_slack(self, solution_samples: torch.Tensor) -> torch.Tensor:

        if not self.is_generated or self.A is None or self.b is None:
            raise RuntimeError("You must call generate() first to generate the constraint matrices A and b.")
        if solution_samples.shape[0] != self.n:
            raise ValueError(f"The variable dimension of the input solution must be {self.n}。")

        LHS = self.A @ solution_samples

        slack = LHS - self.b

        return slack

    def to(self, device: torch.device):
        """Move linear constraint tensors to device."""
        super().to(device)
        if self.A is not None:
            self.A = self.A.to(device)
        if self.b is not None:
            self.b = self.b.to(device)
        return self


