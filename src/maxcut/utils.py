from typing import Any, Dict, Literal
import numpy as np
from itertools import chain, combinations
from dhg import Hypergraph
import torch
from ..utils import compact_vertex_indices
from pyscipopt import quicksum, Model, SCIP_PARAMSETTING, SCIP_EVENTTYPE, Eventhdlr
from ..core import BaseSCIPSolver


def build_hypergraph_maxcut(source_hyperedge_list, renumber_index: bool = True, start_index=0):
    """Construct the structure of the hypergraph maxcut problem

    For the maxcut problem on hypergraph, we have a loss function (see `src/maxcut/loss/loss_max_cut`),
    but we can also construct the maxcut problem on the original hypergraph as an additional hypergraph
    and train it using the pubo loss (see `src/loss/loss_pubo`).
    """
    if renumber_index:
        source_list = compact_vertex_indices(source_hyperedge_list, strat=start_index)
    else:
        source_list = source_hyperedge_list
    max_cut_polynomial = split_polynomial(source_list)
    dst_edges = list(max_cut_polynomial.keys())
    dst_edges_weight = [-1 * x for x in list(max_cut_polynomial.values())]
    dst_hypergraph = Hypergraph(num_v=len(set(np.hstack(source_list))), e_list=dst_edges, e_weight=dst_edges_weight)
    return dst_hypergraph


def generate_subsets(elements: list):
    r"""
    Find all the subsets of k child elements in the set, where k in 1..len(elements)

    Args:
        elements (list): A list containing point indices representing an hyperedge

    Examples:
        >>> print(list(generate_subsets([1, 7, 9])))
        [(1,), (7,), (9,), (1, 7), (1, 9), (7, 9), (1, 7, 9)]

    """
    return chain.from_iterable(combinations(elements, r) for r in range(1, len(elements) + 1))


def split_polynomial(sets: list[list]) -> dict[tuple, int]:
    r"""
    Get the terms of the polynomial and their coefficients in max-cut problem

    Args:
        sets (list[list]): A list containing lists of many x_i, or means a hyperedge list

    Examples:
        >>> print(splitpolynomial([[1, 2, 3]])) # [1, 2, 3] means polynomial variables {x_1, x_2, x_3} or a hyperedge containing {v_1, v_2, v_3}
        {(1,):1, (2,):1, (3,):1, (1, 2):-1, (1, 3):-1, (2, 3):-1}
        it represents a polynomial: x_1 + x_2 + x_3 - x_1x_2 - x_1x_3 - x_2x_3
    """
    coefficients = {}
    for e in sets:
        e_sorted = sorted(e)
        n = len(e_sorted)
        for subset in generate_subsets(e_sorted):
            # k = len(subset)
            # # For $\prod_{i \in e} x_i$: If the length of $e$ is an even number, its coefficient is -2; otherwise, it is 0.
            # is_full_edge = tuple(subset) == tuple(e_sorted)
            # if is_full_edge:
            #     coeff = (-1) ** (n + 1) - 1
            # else:
            #     coeff = (-1) ** (k + 1)

            # if coeff != 0:
            #     term = tuple(sorted(subset))

            k = len(subset)
            if k == n:
                coeff = -2 if n % 2 == 0 else 0
            else:
                coeff = (-1) ** (k + 1)
            term = tuple(subset)
            coefficients[term] = coefficients.get(term, 0) + coeff

    return {term: c for term, c in coefficients.items() if c != 0}


def maxcut_evaluate(
    outs: torch.Tensor, graph, threshold=0.7
) -> Dict[Literal["cut_edges", "total_edges", "accuracy", "group_distribution", "not_converged"], Any]:
    """
    Evaluate the effectiveness of Max-Cut partitioning with [0, 1] outputs (like `sigmoid`)

    Args:
        outs: Model output tensor shape (num_nodes, 1) (sigmoid probabilities for group `1`)
        graph: The graph object
        threshold: Confidence threshold for node convergence

    Returns:
        {
            "cut_edges": Number of successfully cut edges,
            "total_edges": Total number of edges,
            "accuracy": Cut accuracy rate,
            "group_distribution": Tuple of (group0_count, group1_count),
            "not_converged": Number of unconverged nodes
        }
    """
    nodes = list(graph.v)

    edges = graph.e[0]

    node2idx = {v: i for i, v in enumerate(nodes)}

    if outs.shape != (len(nodes), 1):
        raise ValueError(f"Output should have shape [{(len(nodes))}, 1], got {outs.shape}")

    sigmoid_vals = outs.squeeze(1)
    preds = (sigmoid_vals >= 0.5).long()

    confidences = torch.where(preds == 1, sigmoid_vals, 1 - sigmoid_vals)
    not_converged = (confidences < threshold).sum().item()
    group0 = (preds == 0).sum().item()
    group1 = (preds == 1).sum().item()

    cut_edges = 0
    for edge in edges:
        try:
            indices = [node2idx[v] for v in edge]
        except KeyError as e:
            raise ValueError(f"Edge {edge} contains unknown node: {e}")

        # 关键修改2: 检查超边中是否存在至少两个不同分组的节点
        groups = [preds[i].item() for i in indices]
        if len(set(groups)) >= 2:
            cut_edges += 1

    total_edges = len(edges)
    accuracy = cut_edges / total_edges if total_edges > 0 else 0.0

    return {
        "cut_edges": cut_edges,
        "total_edges": total_edges,
        "accuracy": accuracy,
        "group_distribution": (group0, group1),
        "not_converged": not_converged,
    }


class MaxCutSCIPSolver(BaseSCIPSolver):
    def __init__(self, edge_list, pre_solve=True):
        super().__init__(edge_list, pre_solve)
        self.x = None  # Vertices {0,1}
        self.a = None  # a[e] `v` in edge `e` all 0 or not
        self.b = None  # b[e] `v` in edge `e` all 1 or not
        self.z = None  # z[e] `e` is a cut
        self.edges = None

    def _add_variables(self):
        self.x = {v: self.model.addVar(vtype="B", name=f"x_{v}") for v in self.V}
        unique_edges = {tuple(sorted(e)) for e in self.edge_list}
        self.edges = list(unique_edges)
        self.a = {}
        self.b = {}
        self.z = {}
        for e in self.edges:
            e_str = "_".join(map(str, e))
            self.a[e] = self.model.addVar(vtype="B", name=f"a_{e_str}")
            self.b[e] = self.model.addVar(vtype="B", name=f"b_{e_str}")
            self.z[e] = self.model.addVar(vtype="B", name=f"z_{e_str}")

    def _add_constraints(self):
        for e in self.edges:
            # constrain a[e] `v` in edge `e` all 0 or not
            for v in e:
                self.model.addCons(self.x[v] <= 1 - self.a[e], name=f"a_{e}_{v}_leq")
            sum_x = quicksum(self.x[v] for v in e)
            self.model.addCons(sum_x >= 1 - self.a[e], name=f"a_{e}_sum_geq")

            # constrain b[e] `v` in edge `e` all 1 or not
            for v in e:
                self.model.addCons(self.x[v] >= self.b[e], name=f"b_{e}_{v}_geq")
            sum_neg_x = quicksum(1 - self.x[v] for v in e)
            self.model.addCons(sum_neg_x >= 1 - self.b[e], name=f"b_{e}_sum_neg_geq")

            # constrain z[e] <= 1 - a[e] - b[e]
            self.model.addCons(self.z[e] <= 1 - (self.a[e] + self.b[e]), name=f"z_{e}_constraint")

    def _set_objective(self):
        total_cut = quicksum(self.z[e] for e in self.edges)
        self.model.setObjective(total_cut, "maximize")

    def _get_solution_metrics(self, sol):
        cut_count = sum(round(self.model.getSolVal(sol, self.z[e])) for e in self.edges)
        return {"cut_edges": cut_count}

    def _extract_solution(self, sol):
        solution = {v: round(self.model.getSolVal(sol, self.x[v])) for v in self.V}
        return solution
