from abc import ABC, abstractmethod
from typing import List, Literal, Optional
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .loss import loss_pubo


class BaseConstraint(ABC):
    """
    Interface for compute the loss of a specific constraint

    Attributes:
        cons_items (List[int]):
            Indices of variables required to participate in the constraint (loss) calculation.
        coefficient (float):
            Scale of control constraints, or coefficients within the constraint.
        expect (float):
            Tolerance for this constraint (loss)
        name (Optional[str]):
            Just name :P
    """

    def __init__(self, cons_items: List[int] = [], coefficient: float = 1.0, expect=0, name: Optional[str] = None):
        self.cons_items = cons_items
        self.coefficient = coefficient
        self.expect = expect
        self.name = name if name else f"{self.__class__.__name__}_unnamed"

    @abstractmethod
    def compute_loss(self, outs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Abstract method to compute constraint loss."""
        pass


class UpperBoundConstraint(BaseConstraint):
    """
    High-Order Mutex constraint to pubo problem can be expressed as `x_i + x_j + ... <= UpperBound`

    see also :class:`src.constraints.BaseConstraint`
    see also :class:`src.constraints.ConstraintManager`
    """
    def __init__(self, cons_items = [], coefficient = 1, expect=0, cofs=None, upper_bound=1, name = None):
        super().__init__(cons_items, coefficient, expect, name)
        if cofs is None:
            raise ValueError("cofs must")
        self.cofs = cofs
        self.upper_bound=upper_bound
    def compute_loss(self, outs, **kwargs):
        _outs = outs.squeeze(-1)  # tensor([3.9723e-08, 3.8762e-07, 2.0502e-05,...]) shape(n, )
        epoch = kwargs.get("epoch", 1)
        cofs = self.cofs 
        burden = (_outs[self.cons_items] * cofs).sum() - self.upper_bound
        loss = F.relu(burden)
        return self.coefficient * epoch * 0.1 * loss 

class HighOrderMutexConstraint(BaseConstraint):
    """
    High-Order Mutex constraint to pubo problem can be expressed as `a * (x_1x_2x_3...)`

    see also :class:`src.constraints.BaseConstraint`
    see also :class:`src.constraints.ConstraintManager`
    """

    def compute_loss(self, outs, **kwargs):
        epoch = kwargs.get("epoch", 1)
        loss = self.coefficient * outs[self.cons_items].prod(dim=0)
        return epoch * 0.1 * loss


class OneHotConstraint(BaseConstraint):
    """
    One Hot constraint to pubo problem can be expressed as `a * (x_1 + x_2 + x_3 + ...- 1 )^2`

    see also :class:`src.constraints.BaseConstraint`
    see also :class:`src.constraints.ConstraintManager`
    """

    def compute_loss(self, outs, **kwargs):
        epoch = kwargs.get("epoch", 1)
        loss = self.coefficient * (outs[self.cons_items].sum() - torch.tensor(1.0)).pow(2)
        return epoch * 0.1 * loss


class ConstraintManager:
    """
    Responsible for adding and managing constraint items in the pubo problem

    In PUBO (Polynomial Unconstrained Binary Optimization) problems, constraint terms are often indispensable.
    Transform constraint terms into penalty terms and add them to the loss function, we can use neural network to train them.

    You can also let `compute_total_loss(outs)` participate in back propagation.
    After training, `is_valid(outs)` can test whether each constraint is satisfied in the final model output.

    Notes:
        This class depends on the implementation class of `src.constraints.BaseConstraint`, which defines the penalty term and participates in the calculation of the loss.


    Eaxmples:
        >>> cons_manager = ConstraintManager()
        loss_cons = (cons_manager
                    .add_constraint(HighOrderMutexConstraint([0,2,693], 1, "h1"))
                    .add_constraint(OneHotConstraint([1,8,96,2], 30, expect=1))
                    .add_constraint(CustomConstraint([], 1, name="name_3"))
                    .compute_total_loss(outs))
        total_loss = loss_model + loss_cons
        total_loss.backward()
    """

    def __init__(self):
        self._constraints: List[BaseConstraint] = []

    def add_constraint(self, constraint_instance: BaseConstraint) -> "ConstraintManager":
        self._constraints.append(constraint_instance)
        return self

    def compute_total_loss(self, outs: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Sum the loss of all constraints

        Args:
            outs (torch.Tensor): outputs of Net Model.
            kwargs: external parameters dict to the custom constraint loss function.

        Notes:
        -------
            You can use this function to make constraints participate in back-propagation.
        """
        total = torch.zeros(1, device=outs.device)
        for cons_ins in self._constraints:
            total += cons_ins.compute_loss(outs, **kwargs)
        return total

    def clear(self):
        self._constraints.clear()

    def is_valid(self, outs: torch.Tensor, log_level: Literal["debug", "ignore"] = "debug"):
        outs = (outs > 0.5).float()
        all_verified = True
        for cons_ins in self._constraints:
            _cons_loss = cons_ins.compute_loss(outs).item()
            is_satisfied = _cons_loss <= cons_ins.expect

            all_verified &= is_satisfied

            if log_level == "debug":
                stats = "\033[32mPASS\033[0m" if is_satisfied else "\033[31mFAIL\033[0m"
                log_msg = (
                    f"Constraint '{cons_ins.name}' {stats} | "
                    f"Items: {cons_ins.cons_items} | "
                    f"Coefficient: {cons_ins.coefficient:.2f} | "
                    f"Outputs: {(outs[cons_ins.cons_items]).int().squeeze().tolist()} | "
                    f"Expect: {cons_ins.expect} |"
                    f"Actual: {_cons_loss} |"
                )
                print(log_msg)
        return all_verified


def loss_pubo_with_constraints(outs, **kwargs):
    H = kwargs.get("H", None)
    C = kwargs.get("C", None)
    cons_manager: ConstraintManager = kwargs.get("cons_manager", None)

    l_pubo = loss_pubo(outs, H, C)
    l_cons = cons_manager.compute_total_loss(outs, **kwargs)
    epoch = kwargs.get("epoch", 200)
    if epoch % 250 == 0:
        tqdm.write(f"Epoch: {epoch} | " f"Pubo Loss: {l_pubo.item():.4f} | " f"Constraint Loss: {l_cons.item():.8f}")

    return l_pubo + l_cons


def _loss_high_order_mutex_constraint(outs: torch.Tensor, cons_items, coefficients):
    r"""
    Add High-Order Mutex constraint to pubo problem

    High-Order Mutex can be expressed as `a * (x_1x_2x_3...)`.
    """
    loss_cons = torch.zeros(1)
    for items, coe in zip(cons_items, coefficients):
        loss_cons += coe * outs[items].prod(dim=0)
    return loss_cons


def _loss_one_hot_constraint(outs: torch.Tensor, cons_items, coefficients):
    r"""
    Add One Hot constraint to pubo problem

    One Hot constraint can be expressed as `a * (x_1 + x_2 + x_3 + ...- 1 )^2`
    """
    loss_cons = torch.zeros(1)
    for items, coe in zip(cons_items, coefficients):
        loss_cons += coe * (outs[items].sum() - torch.tensor(1)).pow(2)
    return loss_cons
